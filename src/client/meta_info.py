"""
Metadata information for collection fields.
"""
class CollectionFieldNames:
    ID = "_id"
    DOCUMENT = "document"
    EMBEDDING = "embedding"
    METADATA = "metadata"

    ALL_FIELDS = [ID, DOCUMENT, EMBEDDING, METADATA]

class CollectionNames:
    @staticmethod
    def table_name(collection_name: str) -> str:
        return f"c$v1${collection_name}"