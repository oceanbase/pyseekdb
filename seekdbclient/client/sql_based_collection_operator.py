import logging
from typing import Any, Dict, List, Optional, Union
from .meta_info import CollectionFieldNames, CollectionNames
from .sql_utils import SqlStringifier

logger = logging.getLogger(__name__)

class SqlBasedCollectionOperator:
    """
    SQL based collection operator
    """
    @staticmethod
    def add(client: "BaseClient",
            collection_name: str,
            ids: Union[str, List[str]] = None,
            vectors: Optional[Union[List[float], List[List[float]]]] = None,
            metadatas: Optional[Union[Dict, List[Dict]]] = None,
            documents: Optional[Union[str, List[str]]] = None
        ) -> None:
        if not documents and not vectors and not metadatas:
            raise ValueError("at least one of documents, embeddings, or metadatas must be provided")

        if isinstance(ids, str):
            ids = [ids]
        if isinstance(documents, str):
            documents = [documents]
        if isinstance(metadatas, str):
            metadatas = [metadatas]
        if isinstance(vectors, list) and vectors and not isinstance(vectors[0], list):
            vectors = [vectors]
      
        num_item = 0
        if ids:
            num_item = len(ids)
        elif documents:
            num_item = len(documents)
        elif vectors:
            num_item = len(vectors)
        elif metadatas:
            num_item = len(metadatas)

        if num_item == 0:
            raise ValueError("no items to add")

        if ids and len(ids) != num_item:
            raise ValueError(
                f"The number of ids({len(ids)}) is not equal to the number of items({num_item}).")
        if documents and len(documents) != num_item:
            raise ValueError(
                f"The number of documents({len(documents)}) is not equal to the number of items({num_item}).")
        if metadatas and len(metadatas) != num_item:
            raise ValueError(
                f"The number of metadatas({len(metadatas)}) is not equal to the number of items({num_item}).")
        if vectors and len(vectors) != num_item:
            raise ValueError(
                f"The number of vectors({len(vectors)}) is not equal to the number of items({num_item}).")

        sql_stringifier = SqlStringifier()
        fields = [CollectionFieldNames.DOCUMENT, CollectionFieldNames.METADATA, CollectionFieldNames.EMBEDDING]
        if ids:
            fields.append(CollectionFieldNames.ID)
        fields_sql = ','.join(fields)
        base_sql = f'INSERT INTO {CollectionNames.table_name(collection_name)} ({fields_sql}) VALUES '
        values_list = []
        for i in range(num_item):
            document = sql_stringifier.stringify_value(documents[i]) if documents else "NULL"
            metadata = sql_stringifier.stringify_value(metadatas[i]) if metadatas else "NULL"
            embedding = sql_stringifier.stringify_value({vectors[i]}) if vectors else "NULL"
            value_items = [document, metadata, embedding]
            if ids:
                value_items.append(sql_stringifier.stringify_value(ids[i]))
            values_list.append('(' + ','.join(value_items) + ')')
        values_str = ','.join(values_list)
        sql = base_sql + values_str

        logger.debug(f"add data to collection. collection={collection_name}, sql={sql}")
        client.execute(sql)

    @staticmethod
    def update(client: "BaseClient",
               collection_name: str,
               ids: Union[str, List[str]],
               vectors: Optional[Union[List[float], List[List[float]]]] = None,
               metadatas: Optional[Union[Dict, List[Dict]]] = None,
               documents: Optional[Union[str, List[str]]] = None
               ) -> None:
        if isinstance(ids, str):
            ids = [ids]
        if not ids:
            raise ValueError(f"ids must not be empty")
        if isinstance(documents, str):
            documents = [documents]
        if isinstance(metadatas, str):
            metadatas = [metadatas]
        if vectors and isinstance(vectors, list) and not isinstance(vectors[0], list):
            vectors = [vectors]
        if not documents and not metadatas and not vectors:
            raise ValueError(f"You must specify at least one column to update")
        if documents and len(documents) != len(ids):
            raise ValueError(
                f"The number of documents({len(documents)}) is not equal to the number of ids({len(ids)}).")
        if metadatas and len(metadatas) != len(ids):
            raise ValueError(
                f"The number of metadatas({len(metadatas)}) is not equal to the number of ids({len(ids)}).")
        if vectors and len(vectors) != len(ids):
            raise ValueError(
                f"The number of vectors({len(vectors)}) is not equal to the number of ids({len(ids)}).")

        sql_stringifier = SqlStringifier()
        with client.begin():
            base_sql = f"UPDATE {CollectionNames.table_name(collection_name)} SET "
            for i in range(len(ids)):
                where_sql = f" WHERE {CollectionFieldNames.ID}={sql_stringifier.stringify_value(ids[i])}"
                set_sql = ''
                set_sql += f",{CollectionFieldNames.DOCUMENT}={sql_stringifier.stringify_value(documents[i])}" \
                    if documents else ""
                set_sql += f",{CollectionFieldNames.METADATA}={sql_stringifier.stringify_value(metadatas[i])}" \
                    if metadatas else ""
                set_sql += f",{CollectionFieldNames.EMBEDDING}={sql_stringifier.stringify_value(vectors[i])}" \
                    if vectors else ""
                sql = base_sql + set_sql[1:] + where_sql
                logger.debug(f"Update document. collection={collection_name}, sql={sql}")
                client.execute(sql)
    
    @staticmethod
    def upsert(client: "BaseClient",
               collection_name: str,
               ids: Union[str, List[str]],
               vectors: Optional[Union[List[float], List[List[float]]]] = None,
               metadatas: Optional[Union[Dict, List[Dict]]] = None,
               documents: Optional[Union[str, List[str]]] = None
              ) -> None:
        if not ids:
            raise ValueError(f"ids must not be empty")
        if isinstance(documents, str):
            documents = [documents]
        if isinstance(metadatas, str):
            metadatas = [metadatas]
        if vectors and isinstance(vectors, list) and not isinstance(vectors[0], list):
            vectors = [vectors]
        if not documents and not metadatas and not vectors:
            raise ValueError(f"You must specify at least one column to update")
        if documents and len(documents) != len(ids):
            raise ValueError(
                f"The number of documents({len(documents)}) is not equal to the number of ids({len(ids)}).")
        if metadatas and len(metadatas) != len(ids):
            raise ValueError(
                f"The number of metadatas({len(metadatas)}) is not equal to the number of ids({len(ids)}).")
        if vectors and len(vectors) != len(ids):
            raise ValueError(
                f"The number of vectors({len(vectors)}) is not equal to the number of ids({len(ids)}).")

        sql_stringifier = SqlStringifier()
        fields = CollectionFieldNames.ALL_FIELDS
        base_sql = f"INSERT INTO {CollectionNames.table_name(collection_name)} ({','.join(fields)}) VALUES "
        duplicate_key_sql = " ON DUPLICATE KEY UPDATE "
        with client.begin():
            for i in range(len(ids)):
                document_insert = sql_stringifier.stringify_value(documents[i]) if documents else "NULL"
                metadata_insert = sql_stringifier.stringify_value(metadatas[i]) \
                    if metadatas and metadatas[i] else "NULL"
                embedding_insert = sql_stringifier.stringify_value(vectors[i]) if vectors else "NULL"
                values_str = '(' + ','.join([document_insert, metadata_insert, embedding_insert]) + ')'

                update_set_sql = ''
                update_set_sql += f",{CollectionFieldNames.DOCUMENT}={sql_stringifier.stringify_value(documents[i])}" \
                    if documents and documents[i] is not None else ""
                update_set_sql += f",{CollectionFieldNames.METADATA}={sql_stringifier.stringify_value(metadatas[i])}" \
                    if metadatas and metadatas[i] is not None else ""
                update_set_sql += f",{CollectionFieldNames.EMBEDDING}={sql_stringifier.stringify_value(vectors[i])}" \
                    if vectors and vectors[i] is not None else ""

                sql = base_sql + values_str + duplicate_key_sql + update_set_sql
                logger.debug(f"Upsert collection. collection={collection_name}, sql={sql}")
                client.execute(sql)
    
    @staticmethod
    def delete(client: "BaseClient",
               collection_name: str,
               ids: Optional[Union[str, List[str]]] = None,
               where: Optional[Dict[str, Any]] = None,
               where_document: Optional[Dict[str, Any]] = None,
               **kwargs
           ) -> None:
        with client.begin():
            results_to_delete = client._collection_get(collection_name=collection_name, 
                                                      ids=ids,
                                                      where=where,
                                                      where_document=where_document,
                                                      include=[CollectionFieldNames.ID],
                                                      **kwargs)
            if not results_to_delete:
                return
            if CollectionFieldNames.ID not in results_to_delete:
                raise ValueError(f"[Internal Error] {CollectionFieldNames.ID} not found in results_to_delete. collection={collection_name}")

            sql_stringifier = SqlStringifier()
            ids_to_delete = results_to_delete[CollectionFieldNames.ID]
            if not ids_to_delete:
                logger.debug(f"No ids to delete. collection={collection_name}")
                return
            sql = f"DELETE FROM {CollectionNames.table_name(collection_name)} WHERE {CollectionFieldNames.ID} IN ({','.join(map(sql_stringifier.stringify_value, ids_to_delete))})"
            logger.debug(f"Delete collection data. collection={collection_name}, sql={sql}")
            client.execute(sql)