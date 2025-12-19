import pytest
import sys
import os
from pathlib import Path
import re
import random
import uuid
import json


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))



import pyseekdb
from pyseekdb import DefaultEmbeddingFunction, HNSWConfiguration


SERVER_HOST = os.environ.get('SERVER_HOST', '127.0.0.1')
SERVER_PORT = int(os.environ.get('SERVER_PORT', '2881'))
SERVER_USER = os.environ.get('SERVER_USER', 'root')
SERVER_PASSWORD = os.environ.get('SEEKDB_PASSWORD', '')
DEFAULT_DB_NAME = os.environ.get('SEEKDB_DATABASE', 'test')


class UserDefinedEmbeddingFunction:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = None
        self._model_dimensions = {
            'all-MiniLM-L6-v2': 384,
            'all-mpnet-base-v2': 768,
            'paraphrase-MiniLM-L6-v2': 384,
        }


    def _ensure_model_loaded(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
                print(f"✅ 已加载模型: {self.model_name}")
            except ImportError:
                raise ImportError("sentence-transformers 库未安装。请运行: pip install sentence-transformers")
            except Exception as e:
                raise RuntimeError(f"加载模型 {self.model_name} 失败: {e}")


    @property
    def dimension(self) -> int:
        if self._dimension is None:
            if self.model_name in self._model_dimensions:
                self._dimension = self._model_dimensions[self.model_name]
            else:
                self._ensure_model_loaded()
                test_embedding = self._model.encode(["test"], convert_to_numpy=True)
                self._dimension = len(test_embedding[0])
        return self._dimension


    def __call__(self, input):
        self._ensure_model_loaded()
        if isinstance(input, str):
            input = [input]
        if not input:
            return []
        embeddings = self._model.encode(input, convert_to_numpy=True, show_progress_bar=False)
        return [embedding.tolist() for embedding in embeddings]


class TestCollectionGetComplete:
    """Collection Get操作完整测试：覆盖get所有官方场景"""


    def setup_method(self, method=None):
        pass


    def _record_collection_op(self, test_id, op_type, desc, params=None, result=None, success=True, error_msg="", ignore_fields=None):
        return self.result_recorder.record_collection_operation(
            test_id, op_type, desc, params, result, success, error_msg, ignore_fields
        )


    def _safe_delete(self, client, name):
        try:
            client.delete_collection(name)
        except Exception:
            pass


    def _print_get_results(self, results, operation_description):
        """打印get操作结果的详细信息"""
        print(f"\n📊 {operation_description} 返回结果详情:")
        print("-" * 60)
        
        if not results:
            print("❌ 无返回结果")
            return
            
        if "ids" in results:
            print(f"📋 返回记录数量: {len(results['ids'])}")
            print("-" * 40)
            
            for i, result_id in enumerate(results["ids"]):
                print(f"  记录 {i + 1}:")
                print(f"    ID: {result_id}")
                
                if "documents" in results and len(results["documents"]) > i:
                    doc = results["documents"][i]
                    print(f"    文档: {doc[:80]}{'...' if len(doc) > 80 else ''}")
                
                if "metadatas" in results and len(results["metadatas"]) > i:
                    meta = results["metadatas"][i]
                    print(f"    元数据: {meta}")
                
                if "embeddings" in results and len(results["embeddings"]) > i:
                    emb = results["embeddings"][i]
                    print(f"    嵌入维度: {len(emb)}")
                
                print()


    def _validate_get_result_structure(self, result, include_fields=None):
        """验证get返回结果的结构是否符合chromadb格式"""
        if not isinstance(result, dict):
            return False, "结果不是字典类型"
        
        required_keys = ["ids"]
        optional_keys = ["documents", "metadatas", "embeddings"]
        
        # 检查必需键
        for key in required_keys:
            if key not in result:
                return False, f"缺少必需键: {key}"
            if not isinstance(result[key], list):
                return False, f"{key} 不是列表类型"
        
        # 检查可选键
        if include_fields:
            for field in include_fields:
                if field in result:
                    if not isinstance(result[field], list):
                        return False, f"{field} 不是列表类型"
        
        # 检查内部结构一致性
        n_results = len(result["ids"])
        for key in required_keys + (include_fields or []):
            if key in result:
                if len(result[key]) != n_results:
                    return False, f"{key}长度不一致: {len(result[key])} != {n_results}"
        
        return True, "结构验证成功"


    def test_complete_get_operations(self):
        """完整Get测试：覆盖get所有官方场景"""
        client = None
        coll_name = "coll_get_complete"
        try:
            client = pyseekdb.Client(
                host=SERVER_HOST,
                port=SERVER_PORT,
                database=DEFAULT_DB_NAME,
                user=SERVER_USER,
                password=SERVER_PASSWORD
            )
            self._safe_delete(client, coll_name)


            print("=" * 80)
            print("开始Get完整测试 - 覆盖get所有官方场景")
            print("=" * 80)


            ef = DefaultEmbeddingFunction()
            config = HNSWConfiguration(dimension=384, distance='cosine')
            collection = client.create_collection(name=coll_name, configuration=config, embedding_function=ef)
            dim = collection.dimension


            # 准备测试数据
            test_data = [
                {
                    "id": "doc1",
                    "document": "机器学习是人工智能的重要分支",
                    "embedding": [0.1 * i for i in range(dim)],
                    "metadata": {"category": "AI", "score": 95, "tags": ["ml", "ai"], "version": 1}
                },
                {
                    "id": "doc2", 
                    "document": "深度学习在计算机视觉中有广泛应用",
                    "embedding": [0.2 * i for i in range(dim)],
                    "metadata": {"category": "AI", "score": 88, "tags": ["dl", "cv"], "version": 2}
                },
                {
                    "id": "doc3",
                    "document": "自然语言处理研究语言理解技术",
                    "embedding": [0.3 * i for i in range(dim)],
                    "metadata": {"category": "NLP", "score": 92, "tags": ["nlp", "ai"], "version": 1}
                },
                {
                    "id": "doc4",
                    "document": "数据分析需要统计学和编程技能",
                    "embedding": [0.4 * i for i in range(dim)],
                    "metadata": {"category": "Data", "score": 85, "tags": ["stats", "python"], "version": 2}
                },
                {
                    "id": "doc5",
                    "document": "机器学习算法包括监督学习和无监督学习",
                    "embedding": [0.5 * i for i in range(dim)],
                    "metadata": {"category": "AI", "score": 90, "tags": ["ml", "algorithms"], "version": 3}
                },
                {
                    "id": "doc6",
                    "document": "Python是数据科学的主要编程语言",
                    "embedding": [0.6 * i for i in range(dim)],
                    "metadata": {"category": "Programming", "score": 87, "tags": ["python", "coding"], "version": 1}
                }
            ]


            # 添加测试数据
            for data in test_data:
                collection.add(
                    ids=data["id"],
                    documents=data["document"],
                    embeddings=data["embedding"],
                    metadatas=data["metadata"]
                )


            print("✅ 测试数据准备完成")


            # ==================== 基本GET操作测试 ====================
            print("\n🔍 基本GET操作测试")


            # 1. 按单个ID获取
            try:
                print(f"\n✅ 测试: 按单个ID获取")
                results = collection.get(ids="doc1")
                
                # 打印详细结果
                self._print_get_results(results, "按单个ID获取")
                
                structure_ok, structure_msg = self._validate_get_result_structure(results)
                n_returned = len(results["ids"]) if results.get("ids") else 0
                correct_id = n_returned > 0 and results["ids"][0] == "doc1"
                
                success = structure_ok and n_returned == 1 and correct_id
                
                self._record_collection_op("get-single-id", "get", "按单个ID获取",
                    {"ids": "doc1"},
                    {"structure_ok": structure_ok, "n_returned": n_returned, "correct_id": correct_id},
                    success, structure_msg if not structure_ok else f"返回结果验证失败: n={n_returned}, correct_id={correct_id}")
                
                if success:
                    print("✅ 按单个ID获取测试通过")
                else:
                    print("❌ 按单个ID获取测试失败")
                    
            except Exception as e:
                self._record_collection_op("get-single-id", "get", "按单个ID获取异常",
                    {"ids": "doc1"}, None, False, str(e))


            # 2. 按多个ID获取
            try:
                print(f"\n✅ 测试: 按多个ID获取")
                target_ids = ["doc1", "doc3", "doc5"]
                results = collection.get(ids=target_ids)
                
                # 打印详细结果
                self._print_get_results(results, "按多个ID获取")
                
                structure_ok, structure_msg = self._validate_get_result_structure(results)
                n_returned = len(results["ids"]) if results.get("ids") else 0
                
                # 验证返回的ID是否与请求的一致
                ids_correct = set(results.get("ids", [])) == set(target_ids)
                success = structure_ok and n_returned == len(target_ids) and ids_correct
                
                self._record_collection_op("get-multiple-ids", "get", "按多个ID获取",
                    {"ids": target_ids},
                    {"structure_ok": structure_ok, "n_returned": n_returned, "ids_correct": ids_correct},
                    success, structure_msg if not structure_ok else f"ID验证失败: 期望{target_ids}, 实际{results.get('ids', [])}")
                
                if success:
                    print("✅ 按多个ID获取测试通过")
                else:
                    print("❌ 按多个ID获取测试失败")
                    
            except Exception as e:
                self._record_collection_op("get-multiple-ids", "get", "按多个ID获取异常",
                    {"ids": ["doc1", "doc3", "doc5"]}, None, False, str(e))


            # ==================== 元数据筛选测试 ====================
            print("\n📋 元数据筛选测试")


            # 3. 等值筛选（简化形式）
            try:
                print(f"\n✅ 测试: 等值筛选（简化形式）")
                results = collection.get(
                    where={"category": "AI"},
                    limit=10
                )
                
                # 打印详细结果
                self._print_get_results(results, "等值筛选（简化形式）")
                
                structure_ok, structure_msg = self._validate_get_result_structure(results)
                n_returned = len(results["ids"]) if results.get("ids") else 0
                
                # 验证所有结果的category都是AI
                all_ai = True
                if results.get("metadatas"):
                    for meta in results["metadatas"]:
                        if meta.get("category") != "AI":
                            all_ai = False
                            break
                
                success = structure_ok and all_ai
                expected_count = 3  # doc1, doc2, doc5
                count_correct = n_returned == expected_count
                
                self._record_collection_op("get-where-eq-simple", "get", "等值筛选（简化形式）",
                    {"where": {"category": "AI"}, "limit": 10},
                    {"structure_ok": structure_ok, "n_returned": n_returned, "all_ai": all_ai, "count_correct": count_correct},
                    success and count_correct, 
                    structure_msg if not structure_ok else f"筛选验证失败: 返回{n_returned}个AI记录，期望{expected_count}个")
                
                if success and count_correct:
                    print("✅ 等值筛选（简化形式）测试通过")
                else:
                    print("❌ 等值筛选（简化形式）测试失败")
                    
            except Exception as e:
                self._record_collection_op("get-where-eq-simple", "get", "等值筛选（简化形式）异常",
                    {"where": {"category": "AI"}, "limit": 10}, None, False, str(e))


            # 4. 等值筛选（显式运算符）
            try:
                print(f"\n✅ 测试: 等值筛选（显式运算符）")
                results = collection.get(
                    where={"category": {"$eq": "AI"}},
                    limit=10
                )
                
                # 打印详细结果
                self._print_get_results(results, "等值筛选（显式运算符）")
                
                structure_ok, structure_msg = self._validate_get_result_structure(results)
                n_returned = len(results["ids"]) if results.get("ids") else 0
                
                all_ai = True
                if results.get("metadatas"):
                    for meta in results["metadatas"]:
                        if meta.get("category") != "AI":
                            all_ai = False
                            break
                
                success = structure_ok and all_ai
                expected_count = 3
                count_correct = n_returned == expected_count
                
                self._record_collection_op("get-where-eq-explicit", "get", "等值筛选（显式运算符）",
                    {"where": {"category": {"$eq": "AI"}}, "limit": 10},
                    {"structure_ok": structure_ok, "n_returned": n_returned, "all_ai": all_ai, "count_correct": count_correct},
                    success and count_correct,
                    structure_msg if not structure_ok else f"筛选验证失败: 返回{n_returned}个AI记录，期望{expected_count}个")
                
                if success and count_correct:
                    print("✅ 等值筛选（显式运算符）测试通过")
                else:
                    print("❌ 等值筛选（显式运算符）测试失败")
                    
            except Exception as e:
                self._record_collection_op("get-where-eq-explicit", "get", "等值筛选（显式运算符）异常",
                    {"where": {"category": {"$eq": "AI"}}, "limit": 10}, None, False, str(e))


            # 5. 比较运算符筛选
            try:
                print(f"\n✅ 测试: 比较运算符筛选")
                results = collection.get(
                    where={"score": {"$gte": 90}},
                    limit=10
                )
                
                # 打印详细结果
                self._print_get_results(results, "比较运算符筛选")
                
                structure_ok, structure_msg = self._validate_get_result_structure(results)
                n_returned = len(results["ids"]) if results.get("ids") else 0
                
                # 验证所有结果的score >= 90
                all_high_score = True
                if results.get("metadatas"):
                    for meta in results["metadatas"]:
                        if meta.get("score", 0) < 90:
                            all_high_score = False
                            break
                
                success = structure_ok and all_high_score
                expected_count = 3  # doc1(95), doc3(92), doc5(90)
                count_correct = n_returned == expected_count
                
                self._record_collection_op("get-where-gte", "get", "比较运算符筛选",
                    {"where": {"score": {"$gte": 90}}, "limit": 10},
                    {"structure_ok": structure_ok, "n_returned": n_returned, "all_high_score": all_high_score, "count_correct": count_correct},
                    success and count_correct,
                    structure_msg if not structure_ok else f"筛选验证失败: 返回{n_returned}个高分记录，期望{expected_count}个")
                
                if success and count_correct:
                    print("✅ 比较运算符筛选测试通过")
                else:
                    print("❌ 比较运算符筛选测试失败")
                    
            except Exception as e:
                self._record_collection_op("get-where-gte", "get", "比较运算符筛选异常",
                    {"where": {"score": {"$gte": 90}}, "limit": 10}, None, False, str(e))


            # 6. $in运算符筛选
            try:
                print(f"\n✅ 测试: $in运算符筛选")
                results = collection.get(
                    where={"tags": {"$in": ["ml", "python"]}},
                    limit=10
                )
                
                # 打印详细结果
                self._print_get_results(results, "$in运算符筛选")
                
                structure_ok, structure_msg = self._validate_get_result_structure(results)
                n_returned = len(results["ids"]) if results.get("ids") else 0
                
                # 验证所有结果包含ml或python标签
                all_contain_tags = True
                if results.get("metadatas"):
                    for meta in results["metadatas"]:
                        tags = meta.get("tags", [])
                        if "ml" not in tags and "python" not in tags:
                            all_contain_tags = False
                            break
                
                success = structure_ok and all_contain_tags
                expected_count = 4  # doc1(ml), doc4(python), doc5(ml), doc6(python)
                count_correct = n_returned == expected_count
                
                self._record_collection_op("get-where-in", "get", "$in运算符筛选",
                    {"where": {"tags": {"$in": ["ml", "python"]}}, "limit": 10},
                    {"structure_ok": structure_ok, "n_returned": n_returned, "all_contain_tags": all_contain_tags, "count_correct": count_correct},
                    success and count_correct,
                    structure if not structure_ok else f"筛选验证失败: 返回{n_returned}个匹配记录，期望{expected_count}个")
                
                if success and count_correct:
                    print("✅ $in运算符筛选测试通过")
                else:
                    print("❌ $in运算符筛选测试失败")
                    
            except Exception as e:
                self._record_collection_op("get-where-in", "get", "$in运算符筛选异常",
                    {"where": {"tags": {"$in": ["ml", "python"]}}, "limit": 10}, None, False, str(e))


            # 7. 逻辑运算符筛选（$or）
            try:
                print(f"\n✅ 测试: 逻辑运算符筛选（$or）")
                results = collection.get(
                    where={
                        "$or": [
                            {"category": "AI"},
                            {"tags": "python"}
                        ]
                    },
                    limit=10
                )
                
                # 打印详细结果
                self._print_get_results(results, "逻辑运算符筛选（$or）")
                
                structure_ok, structure_msg = self._validate_get_result_structure(results)
                n_returned = len(results["ids"]) if results.get("ids") else 0
                
                # 验证所有结果满足category=AI或包含python标签
                all_match_or = True
                if results.get("metadatas"):
                    for meta in results["metadatas"]:
                        category_ok = meta.get("category") == "AI"
                        tags_ok = "python" in meta.get("tags", [])
                        if not (category_ok or tags_ok):
                            all_match_or = False
                            break
                
                success = structure_ok and all_match_or
                expected_count = 5  # AI: doc1,doc2,doc5; python: doc4,doc6
                count_correct = n_returned == expected_count
                
                self._record_collection_op("get-where-or", "get", "逻辑运算符筛选（$or）",
                    {"where": {"$or": [{"category": "AI"}, {"tags": "python"}]}, "limit": 10},
                    {"structure_ok": structure_ok, "n_returned": n_returned, "all_match_or": all_match_or, "count_correct": count_correct},
                    success and count_correct,
                    structure_msg if not structure_ok else f"筛选验证失败: 返回{n_returned}个匹配记录，期望{expected_count}个")
                
                if success and count_correct:
                    print("✅ 逻辑运算符筛选（$or）测试通过")
                else:
                    print("❌ 逻辑运算符筛选（$or）测试失败")
                    
            except Exception as e:
                self._record_collection_op("get-where-or", "get", "逻辑运算符筛选（$or）异常",
                    {"where": {"$or": [{"category": "AI"}, {"tags": "python"}]}, "limit": 10}, None, False, str(e))


            # ==================== 文档内容筛选测试 ====================
            print("\n📄 文档内容筛选测试")


            # 8. 文档内容包含筛选
            try:
                print(f"\n✅ 测试: 文档内容包含筛选")
                results = collection.get(
                    where_document={"$contains": "机器学习"},
                    limit=10
                )
                
                # 打印详细结果
                self._print_get_results(results, "文档内容包含筛选")
                
                structure_ok, structure_msg = self._validate_get_result_structure(results)
                n_returned = len(results["ids"]) if results.get("ids") else 0
                
                # 验证所有文档包含"机器学习"
                all_contain_keyword = True
                if results.get("documents"):
                    for doc in results["documents"]:
                        if "机器学习" not in doc:
                            all_contain_keyword = False
                            break
                
                success = structure_ok and all_contain_keyword
                expected_count = 2  # doc1, doc5
                count_correct = n_returned == expected_count
                
                self._record_collection_op("get-where-document-contains", "get", "文档内容包含筛选",
                    {"where_document": {"$contains": "机器学习"}, "limit": 10},
                    {"structure_ok": structure_ok, "n_returned": n_returned, "all_contain_keyword": all_contain_keyword, "count_correct": count_correct},
                    success and count_correct,
                    structure_msg if not structure_ok else f"筛选验证失败: 返回{n_returned}个匹配记录，期望{expected_count}个")
                
                if success and count_correct:
                    print("✅ 文档内容包含筛选测试通过")
                else:
                    print("❌ 文档内容包含筛选测试失败")
                    
            except Exception as e:
                self._record_collection_op("get-where-document-contains", "get", "文档内容包含筛选异常",
                    {"where_document": {"$contains": "机器学习"}, "limit": 10}, None, False, str(e))


            # ==================== 组合筛选测试 ====================
            print("\n🔗 组合筛选测试")


            # 9. 元数据和文档内容组合筛选
            try:
                print(f"\n✅ 测试: 组合筛选")
                results = collection.get(
                    where={"category": {"$eq": "AI"}},
                    where_document={"$contains": "学习"},
                    limit=10
                )
                
                # 打印详细结果
                self._print_get_results(results, "组合筛选")
                
                structure_ok, structure_msg = self._validate_get_result_structure(results)
                n_returned = len(results["ids"]) if results.get("ids") else 0
                
                # 验证所有结果同时满足两个条件
                all_match_combined = True
                if results.get("metadatas") and results.get("documents"):
                    for i, meta in enumerate(results["metadatas"]):
                        doc = results["documents"][i]
                        category_ok = meta.get("category") == "AI"
                        doc_ok = "学习" in doc
                        if not (category_ok and doc_ok):
                            all_match_combined = False
                            break
                
                success = structure_ok and all_match_combined
                expected_count = 3  # doc1, doc2, doc5
                count_correct = n_returned == expected_count
                
                self._record_collection_op("get-combined-filters", "get", "组合筛选",
                    {"where": {"category": {"$eq": "AI"}}, "where_document": {"$contains": "学习"}, "limit": 10},
                    {"structure_ok": structure_ok, "n_returned": n_returned, "all_match_combined": all_match_combined, "count_correct": count_correct},
                    success and count_correct,
                    structure_msg if not structure_ok else f"筛选验证失败: 返回{n_returned}个匹配记录，期望{expected_count}个")
                
                if success and count_correct:
                    print("✅ 组合筛选测试通过")
                else:
                    print("❌ 组合筛选测试失败")
                    
            except Exception as e:
                self._record_collection_op("get-combined-filters", "get", "组合筛选异常",
                    {"where": {"category": {"$eq": "AI"}}, "where_document": {"$contains": "学习"}, "limit": 10}, 
                    None, False, str(e))


            # ==================== 分页测试 ====================
            print("\n📄 分页测试")


            # 10. 分页获取
            try:
                print(f"\n✅ 测试: 分页获取")
                # 先获取所有结果
                all_results = collection.get(limit=100)
                total_count = len(all_results["ids"]) if all_results.get("ids") else 0
                
                # 分页获取：limit=2, offset=1
                results = collection.get(limit=2, offset=1)
                
                # 打印详细结果
                self._print_get_results(results, "分页获取")
                
                structure_ok, structure_msg = self._validate_get_result_structure(results)
                n_returned = len(results["ids"]) if results.get("ids") else 0
                
                # 验证分页结果
                pagination_correct = n_returned == 2  # 应该返回2条记录
                success = structure_ok and pagination_correct
                
                self._record_collection_op("get-pagination", "get", "分页获取",
                    {"limit": 2, "offset": 1},
                    {"structure_ok": structure_ok, "n_returned": n_returned, "total_count": total_count, "pagination_correct": pagination_correct},
                    success,
                    structure_msg if not structure_ok else f"分页验证失败: 返回{n_returned}个记录，期望2个")
                
                if success:
                    print("✅ 分页获取测试通过")
                else:
                    print("❌ 分页获取测试失败")
                    
            except Exception as e:
                self._record_collection_op("get-pagination", "get", "分页获取异常",
                    {"limit": 2, "offset": 1}, None, False, str(e))


            # ==================== 包含字段测试 ====================
            print("\n📊 包含字段测试")


            # 11. 包含特定字段
            try:
                print(f"\n✅ 测试: 包含特定字段")
                results = collection.get(
                    ids=["doc1", "doc2"],
                    include=["documents", "metadatas", "embeddings"]
                )
                
                # 打印详细结果
                self._print_get_results(results, "包含特定字段")
                
                structure_ok, structure_msg = self._validate_get_result_structure(results, ["documents", "metadatas", "embeddings"])
                n_returned = len(results["ids"]) if results.get("ids") else 0
                
                # 验证包含的字段
                has_docs = "documents" in results and len(results["documents"]) == 2
                has_metas = "metadatas" in results and len(results["metadatas"]) == 2
                has_embs = "embeddings" in results and len(results["embeddings"]) == 2
                success = structure_ok and has_docs and has_metas and has_embs
                
                self._record_collection_op("get-include-fields", "get", "包含特定字段",
                    {"ids": ["doc1", "doc2"], "include": ["documents", "metadatas", "embeddings"]},
                    {"structure_ok": structure_ok, "has_documents": has_docs, "has_metadatas": has_metas, "has_embeddings": has_embs},
                    success,
                    structure_msg if not structure_ok else f"字段包含验证失败: docs={has_docs}, metas={has_metas}, embs={has_embs}")
                
                if success:
                    print("✅ 包含特定字段测试通过")
                else:
                    print("❌ 包含特定字段测试失败")
                    
            except Exception as e:
                self._record_collection_op("get-include-fields", "get", "包含特定字段异常",
                    {"ids": ["doc1", "doc2"], "include": ["documents", "metadatas", "embeddings"]}, 
                    None, False, str(e))


            # 12. 获取所有数据（带limit）
            try:
                print(f"\n✅ 测试: 获取所有数据")
                results = collection.get(limit=100)
                
                # 打印简要结果（数据多时不打印详情）
                n_returned = len(results["ids"]) if results.get("ids") else 0
                print(f"📋 返回记录总数: {n_returned}")
                
                structure_ok, structure_msg = self._validate_get_result_structure(results)
                success = structure_ok and n_returned == len(test_data)
                
                self._record_collection_op("get-all-data", "get", "获取所有数据",
                    {"limit": 100},
                    {"structure_ok": structure_ok, "n_returned": n_returned, "expected": len(test_data)},
                    success,
                    structure_msg if not structure_ok else f"数据获取失败: 返回{n_returned}个记录，期望{len(test_data)}个")
                
                if success:
                    print("✅ 获取所有数据测试通过")
                else:
                    print("❌ 获取所有数据测试失败")
                    
            except Exception as e:
                self._record_collection_op("get-all-data", "get", "获取所有数据异常",
                    {"limit": 100}, None, False, str(e))


            # ==================== 边界和错误情况测试 ====================
            print("\n⚠️ 边界和错误情况测试")


            # 13. 获取不存在的ID
            try:
                print(f"\n🚫 测试: 获取不存在的ID")
                results = collection.get(ids="nonexistent_id")
                
                # 打印详细结果
                self._print_get_results(results, "获取不存在的ID")
                
                structure_ok, structure_msg = self._validate_get_result_structure(results)
                n_returned = len(results["ids"]) if results.get("ids") else 0
                success = structure_ok and n_returned == 0  # 不存在的ID应该返回0个结果
                
                self._record_collection_op("get-nonexistent-id", "get", "获取不存在的ID",
                    {"ids": "nonexistent_id"},
                    {"structure_ok": structure_ok, "n_returned": n_returned},
                    success,
                    structure_msg if not structure_ok else f"不存在的ID验证失败: 返回{n_returned}个结果，期望0个")
                
                if success:
                    print("✅ 获取不存在的ID测试通过")
                else:
                    print("❌ 获取不存在的ID测试失败")
                    
            except Exception as e:
                self._record_collection_op("get-nonexistent-id", "get", "获取不存在的ID异常",
                    {"ids": "nonexistent_id"}, None, False, str(e))


            # 14. 混合存在和不存在的ID
            try:
                print(f"\n🚫 测试: 混合存在和不存在的ID")
                mixed_ids = ["doc1", "nonexistent_id", "doc3"]
                results = collection.get(ids=mixed_ids)
                
                # 打印详细结果
                self._print_get_results(results, "混合存在和不存在的ID")
                
                structure_ok, structure_msg = self._validate_get_result_structure(results)
                n_returned = len(results["ids"]) if results.get("ids") else 0
                # 应该只返回存在的ID
                success = structure_ok and n_returned == 2 and set(results["ids"]) == {"doc1", "doc3"}
                
                self._record_collection_op("get-mixed-ids", "get", "混合存在和不存在的ID",
                    {"ids": mixed_ids},
                    {"structure_ok": structure_ok, "n_returned": n_returned, "returned_ids": results.get("ids", [])},
                    success,
                    structure_msg if not structure_ok else f"混合ID验证失败: 返回{results.get('ids', [])}，期望['doc1', 'doc3']")
                
                if success:
                    print("✅ 混合存在和不存在的ID测试通过")
                else:
                    print("❌ 混合存在和不存在的ID测试失败")
                    
            except Exception as e:
                self._record_collection_op("get-mixed-ids", "get", "混合存在和不存在的ID异常",
                    {"ids": mixed_ids}, None, False, str(e))


            # 15. 无参数获取（应该返回所有数据或报错）
            try:
                print(f"\n⚠️ 测试: 无参数获取")
                results = collection.get()  # 无参数
                
                # 打印简要结果
                n_returned = len(results["ids"]) if results.get("ids") else 0
                print(f"📋 无参数返回记录数: {n_returned}")
                
                structure_ok, structure_msg = self._validate_get_result_structure(results)
                # 无参数时行为可能不同，这里只验证结构
                success = structure_ok
                
                self._record_collection_op("get-no-params", "get", "无参数获取",
                    {},
                    {"structure_ok": structure_ok, "n_returned": n_returned},
                    success,
                    structure_msg if not structure_ok else "")
                
                if success:
                    print("✅ 无参数获取测试通过")
                else:
                    print("❌ 无参数获取测试失败")
                    
            except Exception as e:
                self._record_collection_op("get-no-params", "get", "无参数获取异常",
                    {}, None, False, str(e))


            # ==================== 最终清理 ====================
            self._safe_delete(client, coll_name)
            
            print("\n" + "=" * 80)
            print("✅ Get完整测试完成！")
            print("=" * 80)


            test_results = self.result_recorder.get_results()
            is_first_run, comparison_passed = self.result_manager.process_test_results(
                test_results, self.has_expected_file
            )
            self.result_recorder.print_summary()
            
        except Exception as e:
            print(f"❌ Get完整测试发生异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if client:
                try:
                    client.close()
                except:
                    pass


    def teardown_method(self):
        print("🧹 测试环境清理完成")


if __name__ == "__main__":
    print("=" * 80)
    print("pyseekdb - Collection Get Complete Tests")
    print("=" * 80)
    print(f"  Server: {SERVER_USER}@{SERVER_HOST}:{SERVER_PORT}, DB={DEFAULT_DB_NAME}")
    print("=" * 80)
    pytest.main([__file__, "-v", "-s"])