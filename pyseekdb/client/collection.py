"""
Collection class - represents a collection and provides unified data operation interface

Design Pattern:
1. Collection itself contains no business logic
2. All operations are delegated to the client that created it
3. Different clients can have completely different underlying implementations
4. User-facing interface is completely consistent
"""
from typing import Any, List, Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .embedding_function import EmbeddingFunction, Documents as EmbeddingDocuments

from .query_result import QueryResult


class Collection:
    """
    Collection unified interface class
    
    Design Principles:
    - Collection is a lightweight wrapper that only holds metadata
    - All operations delegate to the client via self._client._collection_*() methods
    - Different clients (OceanBase, Seekdb, Milvus, etc.) provide different implementations
    - Users see identical interface regardless of which client created the collection
    """
    
    def __init__(
        self,
        client: Any,  # BaseClient instance
        name: str,
        collection_id: Optional[str] = None,
        dimension: Optional[int] = None,
        embedding_function: Optional["EmbeddingFunction[EmbeddingDocuments]"] = None,
        **metadata
    ):
        """
        Initialize collection object
        
        Args:
            client: The client instance that created this collection
            name: Collection name
            collection_id: Collection unique identifier (some databases may need this)
            dimension: Vector dimension
            embedding_function: Embedding function to convert documents to vectors
            **metadata: Other metadata
        """
        self._client = client  # Core: hold reference to the client
        self._name = name
        self._id = collection_id
        self._dimension = dimension
        self._embedding_function = embedding_function
        self._metadata = metadata
    
    # ==================== Properties ====================
    
    @property
    def name(self) -> str:
        """Collection name"""
        return self._name
    
    @property
    def id(self) -> Optional[str]:
        """Collection ID"""
        return self._id
    
    @property
    def dimension(self) -> Optional[int]:
        """Vector dimension"""
        return self._dimension
    
    @property
    def client(self) -> Any:
        """Associated client"""
        return self._client
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Collection metadata"""
        return self._metadata
    
    @property
    def embedding_function(self) -> Optional["EmbeddingFunction[EmbeddingDocuments]"]:
        """Embedding function for this collection"""
        return self._embedding_function
    
    def __repr__(self) -> str:
        return f"Collection(name='{self._name}', dimension={self._dimension}, client={self._client.mode})"
    
    # ==================== DML Operations ====================
    # All methods delegate to client's internal implementation
    
    def add(
        self,
        ids: Union[str, List[str]],
        vectors: Optional[Union[List[float], List[List[float]]]] = None,
        metadatas: Optional[Union[Dict, List[Dict]]] = None,
        documents: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> None:
        """
        Add data to collection
        
        Args:
            ids: Single ID or list of IDs
            vectors: Single vector or list of vectors (optional if documents provided and embedding_function is set)
            metadatas: Single metadata dict or list of metadata dicts (optional)
            documents: Single document or list of documents (optional)
                       If provided without vectors, embedding_function will be used to generate vectors
            **kwargs: Additional parameters
            
        Examples:
            # Add single item with vectors
            collection.add(ids="1", vectors=[0.1, 0.2, 0.3], metadatas={"tag": "A"})
            
            # Add multiple items with vectors
            collection.add(
                ids=["1", "2", "3"],
                vectors=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                metadatas=[{"tag": "A"}, {"tag": "B"}, {"tag": "C"}]
            )
            
            # Add items with documents (vectors will be auto-generated if embedding_function is set)
            collection.add(
                ids=["1", "2"],
                documents=["Hello world", "How are you?"],
                metadatas=[{"tag": "A"}, {"tag": "B"}]
            )
        """
        return self._client._collection_add(
            collection_id=self._id,
            collection_name=self._name,
            ids=ids,
            vectors=vectors,
            metadatas=metadatas,
            documents=documents,
            embedding_function=self._embedding_function,
            **kwargs
        )
    
    def update(
        self,
        ids: Union[str, List[str]],
        vectors: Optional[Union[List[float], List[List[float]]]] = None,
        metadatas: Optional[Union[Dict, List[Dict]]] = None,
        documents: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> None:
        """
        Update existing data in collection
        
        Args:
            ids: Single ID or list of IDs to update
            vectors: New vectors (optional)
            metadatas: New metadata (optional)
            documents: New documents (optional)
            **kwargs: Additional parameters
            
        Note:
            IDs must exist, otherwise an error will be raised
            
        Examples:
            # Update single item
            collection.update(ids="1", metadatas={"tag": "B"})
            
            # Update multiple items
            collection.update(
                ids=["1", "2"],
                vectors=[[0.9, 0.8], [0.7, 0.6]]
            )
        """
        return self._client._collection_update(
            collection_id=self._id,
            collection_name=self._name,
            ids=ids,
            vectors=vectors,
            metadatas=metadatas,
            documents=documents,
            embedding_function=self._embedding_function,
            **kwargs
        )
    
    def upsert(
        self,
        ids: Union[str, List[str]],
        vectors: Optional[Union[List[float], List[List[float]]]] = None,
        metadatas: Optional[Union[Dict, List[Dict]]] = None,
        documents: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> None:
        """
        Insert or update data in collection
        
        Args:
            ids: Single ID or list of IDs
            vectors: Vectors (optional if documents provided)
            metadatas: Metadata (optional)
            documents: Documents (optional)
            **kwargs: Additional parameters
            
        Note:
            If ID exists, update it; otherwise, insert new data
            
        Examples:
            # Upsert single item
            collection.upsert(ids="1", vectors=[0.1, 0.2], metadatas={"tag": "A"})
            
            # Upsert multiple items
            collection.upsert(
                ids=["1", "2", "3"],
                vectors=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            )
        """
        return self._client._collection_upsert(
            collection_id=self._id,
            collection_name=self._name,
            ids=ids,
            vectors=vectors,
            metadatas=metadatas,
            documents=documents,
            embedding_function=self._embedding_function,
            **kwargs
        )
    
    def delete(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Delete data from collection
        
        Args:
            ids: Single ID or list of IDs to delete (optional)
            where: Filter condition on metadata (optional)
            where_document: Filter condition on documents (optional)
            **kwargs: Additional parameters
            
        Note:
            At least one of ids, where, or where_document must be provided
            
        Examples:
            # Delete by IDs
            collection.delete(ids=["1", "2", "3"])
            
            # Delete by metadata filter
            collection.delete(where={"tag": "A"})
            
            # Delete by document filter
            collection.delete(where_document={"$contains": "keyword"})
        """
        return self._client._collection_delete(
            collection_id=self._id,
            collection_name=self._name,
            ids=ids,
            where=where,
            where_document=where_document,
            **kwargs
        )
    
    # ==================== DQL Operations ====================
    
    def query(
        self,
        query_embeddings: Optional[Union[List[float], List[List[float]]]] = None,
        query_texts: Optional[Union[str, List[str]]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> Union[QueryResult, List[QueryResult]]:
        """
        Query collection by vector similarity
        
        Args:
            query_embeddings: Query vector(s) (optional if query_texts provided)
            query_texts: Query text(s) to be embedded (optional if query_embeddings provided)
            n_results: Number of results to return (default: 10)
            where: Filter condition on metadata supporting:
                   - Comparison operators: $eq, $lt, $gt, $lte, $gte, $ne, $in, $nin
                   - Logical operators: $or, $and, $not
            where_document: Filter condition on documents supporting:
                   - $contains: full-text search
                   - $regex: regular expression matching
                   - Logical operators: $or, $and
            include: Fields to include in results, e.g., ["documents", "metadatas", "embeddings"] (optional)
                     By default, returns "documents" and "metadatas". Always includes "_id".
            **kwargs: Additional parameters
            
        Returns:
            - If single vector/text provided: QueryResult object containing query results
            - If multiple vectors/texts provided: List of QueryResult objects, one for each query vector
            Each QueryResult item contains:
            - _id: record ID (always included)
            - document: document text (if included)
            - embedding: vector embedding (if included)
            - metadata: metadata dictionary (if included)
            - distance: similarity distance (always included for query)
            
        Examples:
            # Query by single embedding (returns QueryResult)
            results = collection.query(
                query_embeddings=[0.1, 0.2, 0.3],
                n_results=5
            )
            
            # Query by multiple embeddings (returns List[QueryResult])
            results = collection.query(
                query_embeddings=[[11.1, 12.1, 13.1], [1.1, 2.3, 3.2]],
                n_results=5
            )
            # results[0] is QueryResult for first vector, results[1] for second vector
            
            # Query with filters
            results = collection.query(
                query_embeddings=[[0.1, 0.2, 0.3]],
                where={"chapter": {"$gte": 3}},
                where_document={"$contains": "machine learning"},
                include=["documents", "metadatas", "embeddings"]
            )
            
            # Query by texts (will be embedded automatically)
            results = collection.query(
                query_texts=["my query text"],
                n_results=10
            )
            
            # Query by multiple texts (returns List[QueryResult])
            results = collection.query(
                query_texts=["text1", "text2"],
                n_results=10
            )
        """
        return self._client._collection_query(
            collection_id=self._id,
            collection_name=self._name,
            query_embeddings=query_embeddings,
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
            embedding_function=self._embedding_function,
            **kwargs
        )
    
    def get(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> Union[QueryResult, List[QueryResult]]:
        """
        Get data from collection by IDs or filters
        
        Args:
            ids: Single ID or list of IDs to retrieve (optional)
            where: Filter condition on metadata (optional)
            where_document: Filter condition on documents (optional)
            limit: Maximum number of results to return (optional)
            offset: Number of results to skip (optional)
            include: Fields to include in results, e.g., ["metadatas", "documents", "embeddings"] (optional)
            **kwargs: Additional parameters
            
        Returns:
            - If single ID provided: QueryResult object containing get results for that ID
            - If multiple IDs provided: List of QueryResult objects, one for each ID
            - If filters provided (no IDs): QueryResult object containing all matching results
            
        Note:
            If no parameters provided, returns all data (up to limit)
            
        Examples:
            # Get by single ID (returns QueryResult)
            results = collection.get(ids="1")
            
            # Get by multiple IDs (returns List[QueryResult])
            results = collection.get(ids=["1", "2", "3"])
            # results[0] is QueryResult for ID "1", results[1] for ID "2", etc.
            
            # Get by filter (returns QueryResult)
            results = collection.get(
                where={"tag": "A"},
                limit=10
            )
            
            # Get all data
            results = collection.get(limit=100)
        """
        return self._client._collection_get(
            collection_id=self._id,
            collection_name=self._name,
            ids=ids,
            where=where,
            where_document=where_document,
            limit=limit,
            offset=offset,
            include=include,
            **kwargs
        )
    
    def hybrid_search(
        self,
        query: Optional[Dict[str, Any]] = None,
        knn: Optional[Dict[str, Any]] = None,
        rank: Optional[Dict[str, Any]] = None,
        n_results: int = 10,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Hybrid search combining full-text search and vector similarity search
        
        Args:
            query: Full-text search configuration dict with:
                - where_document: Document filter conditions (e.g., {"$contains": "text"})
                - where: Metadata filter conditions (e.g., {"page": {"$gte": 5}})
                - n_results: Number of results for full-text search (optional)
            knn: Vector search configuration dict with:
                - query_texts: Query text(s) to be embedded (optional if query_embeddings provided)
                - query_embeddings: Query vector(s) (optional if query_texts provided)
                - where: Metadata filter conditions (optional)
                - n_results: Number of results for vector search (optional)
            rank: Ranking configuration dict (e.g., {"rrf": {"rank_window_size": 60, "rank_constant": 60}})
            n_results: Final number of results to return after ranking (default: 10)
            include: Fields to include in results (e.g., ["documents", "metadatas", "embeddings"])
            **kwargs: Additional parameters
            
        Returns:
            Search results dictionary containing ids, distances, metadatas, documents, embeddings, etc.
            
        Examples:
            # Hybrid search with both full-text and vector search
            results = collection.hybrid_search(
                query={
                    "where_document": {"$contains": "machine learning"},
                    "where": {"category": {"$eq": "science"}},
                    "n_results": 10
                },
                knn={
                    "query_texts": ["AI research"],
                    "where": {"year": {"$gte": 2020}},
                    "n_results": 10
                },
                rank={"rrf": {}},
                n_results=5,
                include=["documents", "metadatas", "embeddings"]
            )
        """
        return self._client._collection_hybrid_search(
            collection_id=self._id,
            collection_name=self._name,
            query=query,
            knn=knn,
            rank=rank,
            n_results=n_results,
            include=include,
            embedding_function=self._embedding_function,
            **kwargs
        )
    
    # ==================== Collection Info ====================
    
    def count(self) -> int:
        """
        Get the number of items in collection
        
        Returns:
            Item count
            
        Examples:
            count = collection.count()
            print(f"Collection has {count} items")
        """
        return self._client._collection_count(
            collection_id=self._id,
            collection_name=self._name
        )
    
    def peek(self, limit: int = 10) -> QueryResult:
        """
        Quickly preview the first few items in the collection
        
        Args:
            limit: Number of items to preview (default: 10)
            
        Returns:
            QueryResult object containing the first limit items
            
        Examples:
            # Preview first 5 items
            preview = collection.peek(limit=5)
            for item in preview:
                print(f"ID: {item._id}, Document: {item.document}")
        """
        return self._client._collection_get(
            collection_id=self._id,
            collection_name=self._name,
            limit=limit,
            offset=0,
            include=["documents", "metadatas", "embeddings"]
        )