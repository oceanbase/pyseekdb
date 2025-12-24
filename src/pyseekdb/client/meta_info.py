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
    # Version prefix for collection tables
    _PREFIX = "c$v1$"
    
    @staticmethod
    def table_name(collection_name: str) -> str:
        """Convert collection name to table name."""
        return f"{CollectionNames._PREFIX}{collection_name}"
    
    @staticmethod
    def collection_name(table_name: str) -> str:
        """Extract collection name from table name."""
        if table_name.startswith(CollectionNames._PREFIX):
            return table_name[len(CollectionNames._PREFIX):]
        return table_name
    
    @staticmethod
    def is_collection_table(table_name: str) -> bool:
        """Check if a table name is a collection table."""
        return table_name.startswith(CollectionNames._PREFIX)
    
    @staticmethod
    def table_pattern() -> str:
        """Get SQL LIKE pattern for collection tables."""
        return f"{CollectionNames._PREFIX}%"
    
    @staticmethod
    def prefix() -> str:
        """Get the collection table prefix."""
        return CollectionNames._PREFIX