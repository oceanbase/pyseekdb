"""
Query result wrapper class with JSON serialization support
"""
import json
from typing import Any, Dict, List, Optional


class QueryResultItem:
    """Single query result item"""
    
    def __init__(
        self,
        id: Any,
        document: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        distance: Optional[float] = None
    ):
        """
        Initialize a query result item
        
        Args:
            id: Record ID
            document: Document text (optional)
            embedding: Vector embedding (optional)
            metadata: Metadata dictionary (optional)
            distance: Distance/similarity score (optional)
        """
        self._id = id
        self.document = document
        self.embedding = embedding
        self.metadata = metadata if metadata is not None else {}
        self.distance = distance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {"_id": self._id}
        
        if self.document is not None:
            result["document"] = self.document
        
        if self.embedding is not None:
            result["embedding"] = self.embedding
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        if self.distance is not None:
            result["distance"] = self.distance
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def __repr__(self) -> str:
        return f"QueryResultItem(id={self._id}, distance={self.distance})"


class QueryResult:
    """Query result wrapper with multiple items"""
    
    def __init__(self, items: Optional[List[QueryResultItem]] = None):
        """
        Initialize query result
        
        Args:
            items: List of QueryResultItem objects (optional)
        """
        self.items = items if items is not None else []
    
    def add_item(
        self,
        id: Any,
        document: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        distance: Optional[float] = None
    ) -> None:
        """
        Add a result item
        
        Args:
            id: Record ID
            document: Document text (optional)
            embedding: Vector embedding (optional)
            metadata: Metadata dictionary (optional)
            distance: Distance/similarity score (optional)
        """
        item = QueryResultItem(
            id=id,
            document=document,
            embedding=embedding,
            metadata=metadata,
            distance=distance
        )
        self.items.append(item)
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries"""
        return [item.to_dict() for item in self.items]
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_list(), ensure_ascii=False, indent=2)
    
    def __len__(self) -> int:
        """Return number of items"""
        return len(self.items)
    
    def __getitem__(self, index: int) -> QueryResultItem:
        """Get item by index"""
        return self.items[index]
    
    def __iter__(self):
        """Iterate over items"""
        return iter(self.items)
    
    def __repr__(self) -> str:
        return f"QueryResult(items={len(self.items)})"

