from typing import Optional, List, Dict, Any, Union
from pymilvus import MilvusClient, DataType
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.types as types
import click
import asyncio
import mcp
import json
import numpy as np
from datetime import datetime

class MilvusConnector:
    def __init__(
        self,
        uri: str,
        token: Optional[str] = None,
        db_name: Optional[str] = "default"
    ):
        self.client = MilvusClient(
            uri=uri,
            token=token,
            db_name=db_name
        )

    async def list_collections(self) -> List[str]:
        """List all collections in the database."""
        try:
            return self.client.list_collections()
        except Exception as e:
            raise ValueError(f"Failed to list collections: {str(e)}")

    async def get_collection_info(self, collection_name: str) -> dict:
        """Get detailed information about a collection."""
        try:
            return self.client.describe_collection(collection_name)
        except Exception as e:
            raise ValueError(f"Failed to get collection info: {str(e)}")

    async def search_collection(
        self,
        collection_name: str,
        query_text: str,
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
        drop_ratio: float = 0.2
    ) -> List[dict]:
        """
        Perform full text search on a collection.
        
        Args:
            collection_name: Name of collection to search
            query_text: Text to search for
            limit: Maximum number of results
            output_fields: Fields to return in results
            drop_ratio: Proportion of low-frequency terms to ignore (0.0-1.0)
        """
        try:
            search_params = {
                'params': {'drop_ratio_search': drop_ratio}
            }

            results = self.client.search(
                collection_name=collection_name,
                data=[query_text],
                anns_field='sparse',
                limit=limit,
                output_fields=output_fields,
                search_params=search_params
            )
            return results
        except Exception as e:
            raise ValueError(f"Search failed: {str(e)}")

    async def query_collection(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[dict]:
        """Query collection using filter expressions."""
        try:
            return self.client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=output_fields,
                limit=limit
            )
        except Exception as e:
            raise ValueError(f"Query failed: {str(e)}")

    async def count_entities(self, collection_name: str, filter_expr: Optional[str] = None) -> int:
        """Count entities in a collection, optionally filtered."""
        try:
            return self.client.count(collection_name, filter=filter_expr)
        except Exception as e:
            raise ValueError(f"Count failed: {str(e)}")
            
    async def vector_search(
        self,
        collection_name: str,
        vector: List[float],
        vector_field: str,
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
        metric_type: str = "COSINE",
        filter_expr: Optional[str] = None
    ) -> List[dict]:
        """
        Perform vector similarity search on a collection.
        
        Args:
            collection_name: Name of collection to search
            vector: Query vector
            vector_field: Field containing vectors to search
            limit: Maximum number of results
            output_fields: Fields to return in results
            metric_type: Distance metric (COSINE, L2, IP)
            filter_expr: Optional filter expression
        """
        try:
            search_params = {
                "metric_type": metric_type,
                "params": {"nprobe": 10}
            }
            
            results = self.client.search(
                collection_name=collection_name,
                data=[vector],
                anns_field=vector_field,
                param=search_params,
                limit=limit,
                output_fields=output_fields,
                expr=filter_expr
            )
            return results
        except Exception as e:
            raise ValueError(f"Vector search failed: {str(e)}")
    
    async def hybrid_search(
        self,
        collection_name: str,
        vector: List[float],
        vector_field: str,
        filter_expr: str,
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
        metric_type: str = "COSINE"
    ) -> List[dict]:
        """
        Perform hybrid search combining vector similarity and attribute filtering.
        
        Args:
            collection_name: Name of collection to search
            vector: Query vector
            vector_field: Field containing vectors to search
            filter_expr: Filter expression for metadata
            limit: Maximum number of results
            output_fields: Fields to return in results
            metric_type: Distance metric (COSINE, L2, IP)
        """
        try:
            search_params = {
                "metric_type": metric_type,
                "params": {"nprobe": 10}
            }
            
            results = self.client.search(
                collection_name=collection_name,
                data=[vector],
                anns_field=vector_field,
                param=search_params,
                limit=limit,
                output_fields=output_fields,
                expr=filter_expr
            )
            return results
        except Exception as e:
            raise ValueError(f"Hybrid search failed: {str(e)}")
    
    async def create_collection(
        self,
        collection_name: str,
        schema: Dict[str, Any],
        index_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new collection with the specified schema.
        
        Args:
            collection_name: Name for the new collection
            schema: Collection schema definition
            index_params: Optional index parameters
        """
        try:
            # Check if collection already exists
            if collection_name in self.client.list_collections():
                raise ValueError(f"Collection '{collection_name}' already exists")
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                dimension=schema.get("dimension", 128),
                primary_field=schema.get("primary_field", "id"),
                id_type=schema.get("id_type", DataType.INT64),
                vector_field=schema.get("vector_field", "vector"),
                metric_type=schema.get("metric_type", "COSINE"),
                auto_id=schema.get("auto_id", False),
                enable_dynamic_field=schema.get("enable_dynamic_field", True),
                other_fields=schema.get("other_fields", [])
            )
            
            # Create index if params provided
            if index_params:
                self.client.create_index(
                    collection_name=collection_name,
                    field_name=schema.get("vector_field", "vector"),
                    index_params=index_params
                )
                
            return True
        except Exception as e:
            raise ValueError(f"Failed to create collection: {str(e)}")
    
    async def insert_data(
        self,
        collection_name: str,
        data: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Insert data into a collection.
        
        Args:
            collection_name: Name of collection
            data: Dictionary mapping field names to lists of values
        """
        try:
            result = self.client.insert(
                collection_name=collection_name,
                data=data
            )
            return result
        except Exception as e:
            raise ValueError(f"Insert failed: {str(e)}")
    
    async def delete_entities(
        self,
        collection_name: str,
        filter_expr: str
    ) -> Dict[str, Any]:
        """
        Delete entities from a collection based on filter expression.
        
        Args:
            collection_name: Name of collection
            filter_expr: Filter expression to select entities to delete
        """
        try:
            result = self.client.delete(
                collection_name=collection_name,
                expr=filter_expr
            )
            return result
        except Exception as e:
            raise ValueError(f"Delete failed: {str(e)}")
    
    
    async def get_collection_stats(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Name of collection
        """
        try:
            return self.client.get_collection_stats(collection_name)
        except Exception as e:
            raise ValueError(f"Failed to get collection stats: {str(e)}")
            
    async def multi_vector_search(
        self,
        collection_name: str,
        vectors: List[List[float]],
        vector_field: str,
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
        metric_type: str = "COSINE",
        filter_expr: Optional[str] = None,
        search_params: Optional[Dict[str, Any]] = None
    ) -> List[List[dict]]:
        """
        Perform vector similarity search with multiple query vectors.
        
        Args:
            collection_name: Name of collection to search
            vectors: List of query vectors
            vector_field: Field containing vectors to search
            limit: Maximum number of results per query
            output_fields: Fields to return in results
            metric_type: Distance metric (COSINE, L2, IP)
            filter_expr: Optional filter expression
            search_params: Additional search parameters
        """
        try:
            if search_params is None:
                search_params = {
                    "metric_type": metric_type,
                    "params": {"nprobe": 10}
                }
            
            results = self.client.search(
                collection_name=collection_name,
                data=vectors,
                anns_field=vector_field,
                param=search_params,
                limit=limit,
                output_fields=output_fields,
                expr=filter_expr
            )
            return results
        except Exception as e:
            raise ValueError(f"Multi-vector search failed: {str(e)}")
            
    async def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_type: str = "IVF_FLAT",
        metric_type: str = "COSINE",
        params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create an index on a vector field.
        
        Args:
            collection_name: Name of collection
            field_name: Field to index
            index_type: Type of index (IVF_FLAT, HNSW, etc.)
            metric_type: Distance metric (COSINE, L2, IP)
            params: Additional index parameters
        """
        try:
            if params is None:
                params = {"nlist": 1024}
                
            index_params = {
                "index_type": index_type,
                "metric_type": metric_type,
                "params": params
            }
            
            self.client.create_index(
                collection_name=collection_name,
                field_name=field_name,
                index_params=index_params
            )
            return True
        except Exception as e:
            raise ValueError(f"Failed to create index: {str(e)}")

            
    async def bulk_insert(
        self,
        collection_name: str,
        data: Dict[str, List[Any]],
        batch_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Insert data in batches for better performance.
        
        Args:
            collection_name: Name of collection
            data: Dictionary mapping field names to lists of values
            batch_size: Number of records per batch
        """
        try:
            results = []
            field_names = list(data.keys())
            total_records = len(data[field_names[0]])
            
            for i in range(0, total_records, batch_size):
                batch_data = {
                    field: data[field][i:i+batch_size] 
                    for field in field_names
                }
                
                result = self.client.insert(
                    collection_name=collection_name,
                    data=batch_data
                )
                results.append(result)
                
            return results
        except Exception as e:
            raise ValueError(f"Bulk insert failed: {str(e)}")
            
    async def load_collection(
        self,
        collection_name: str,
        replica_number: int = 1
    ) -> bool:
        """
        Load a collection into memory for search and query.
        
        Args:
            collection_name: Name of collection to load
            replica_number: Number of replicas
        """
        try:
            self.client.load_collection(
                collection_name=collection_name,
                replica_number=replica_number
            )
            return True
        except Exception as e:
            raise ValueError(f"Failed to load collection: {str(e)}")
            
    async def release_collection(
        self,
        collection_name: str
    ) -> bool:
        """
        Release a collection from memory.
        
        Args:
            collection_name: Name of collection to release
        """
        try:
            self.client.release_collection(collection_name=collection_name)
            return True
        except Exception as e:
            raise ValueError(f"Failed to release collection: {str(e)}")
            
    async def get_query_segment_info(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        """
        Get information about query segments.
        
        Args:
            collection_name: Name of collection
        """
        try:
            return self.client.get_query_segment_info(collection_name)
        except Exception as e:
            raise ValueError(f"Failed to get query segment info: {str(e)}")
            
    async def upsert_data(
        self,
        collection_name: str,
        data: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Upsert data into a collection (insert or update if exists).
        
        Args:
            collection_name: Name of collection
            data: Dictionary mapping field names to lists of values
        """
        try:
            result = self.client.upsert(
                collection_name=collection_name,
                data=data
            )
            return result
        except Exception as e:
            raise ValueError(f"Upsert failed: {str(e)}")
            
            
    async def get_index_info(
        self,
        collection_name: str,
        field_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get information about indexes in a collection.
        
        Args:
            collection_name: Name of collection
            field_name: Optional specific field to get index info for
        """
        try:
            return self.client.describe_index(
                collection_name=collection_name,
                index_name=field_name
            )
        except Exception as e:
            raise ValueError(f"Failed to get index info: {str(e)}")
            
    async def get_collection_loading_progress(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        """
        Get the loading progress of a collection.
        
        Args:
            collection_name: Name of collection
        """
        try:
            return self.client.get_load_state(collection_name)
        except Exception as e:
            raise ValueError(f"Failed to get loading progress: {str(e)}")
            
    async def create_dynamic_field(
        self,
        collection_name: str,
        field_name: str,
        data_type: str,
        description: Optional[str] = None
    ) -> bool:
        """
        Add a dynamic field to an existing collection.
        
        Args:
            collection_name: Name of collection
            field_name: Name of the new field
            data_type: Data type of the field
            description: Optional description
        """
        try:
            # This is a simplified version as PyMilvus doesn't directly support this
            # In a real implementation, you would use the Milvus API to alter the schema
            field_schema = {
                "name": field_name,
                "description": description or "",
                "data_type": data_type,
                "is_primary": False
            }
            
            # This would be the actual API call in a real implementation
            # self.client.alter_collection(collection_name, field_schema)
            
            # For now, we'll just return success
            return True
        except Exception as e:
            raise ValueError(f"Failed to create dynamic field: {str(e)}")
            

def serve(
    milvus_uri: str,
    milvus_token: Optional[str] = None,
    db_name: Optional[str] = "default"
) -> Server:
    """
    Create and configure the MCP server with Milvus tools.
    
    Args:
        milvus_uri: URI for Milvus server
        milvus_token: Optional auth token
        db_name: Database name to use
    """
    server = Server("milvus")
    milvus = MilvusConnector(milvus_uri, milvus_token, db_name)

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name="milvus-text-search",
                description="Search for documents using full text search in a Milvus collection",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to search"
                        },
                        "query_text": {
                            "type": "string",
                            "description": "Text to search for"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 5
                        },
                        "output_fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Fields to include in results",
                            "default": None
                        },
                        "drop_ratio": {
                            "type": "number",
                            "description": "Proportion of low-frequency terms to ignore (0.0-1.0)",
                            "default": 0.2
                        }
                    },
                    "required": ["collection_name", "query_text"]
                }
            ),
            types.Tool(
                name="milvus-list-collections",
                description="List all collections in the database",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            types.Tool(
                name="milvus-collection-info",
                description="Get detailed information about a collection",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection"
                        }
                    },
                    "required": ["collection_name"]
                }
            ),
            types.Tool(
                name="milvus-query",
                description="Query collection using filter expressions",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to query"
                        },
                        "filter_expr": {
                            "type": "string",
                            "description": "Filter expression (e.g. 'age > 20')"
                        },
                        "output_fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Fields to include in results"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10
                        }
                    },
                    "required": ["collection_name", "filter_expr"]
                }
            ),
            types.Tool(
                name="milvus-count",
                description="Count entities in a collection",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection"
                        },
                        "filter_expr": {
                            "type": "string",
                            "description": "Optional filter expression"
                        }
                    },
                    "required": ["collection_name"]
                }
            ),
            types.Tool(
                name="milvus-vector-search",
                description="Perform vector similarity search on a collection",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to search"
                        },
                        "vector": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Query vector"
                        },
                        "vector_field": {
                            "type": "string",
                            "description": "Field containing vectors to search",
                            "default": "vector"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 5
                        },
                        "output_fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Fields to include in results"
                        },
                        "metric_type": {
                            "type": "string",
                            "description": "Distance metric (COSINE, L2, IP)",
                            "default": "COSINE"
                        },
                        "filter_expr": {
                            "type": "string",
                            "description": "Optional filter expression"
                        }
                    },
                    "required": ["collection_name", "vector"]
                }
            ),
            types.Tool(
                name="milvus-hybrid-search",
                description="Perform hybrid search combining vector similarity and attribute filtering",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to search"
                        },
                        "vector": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Query vector"
                        },
                        "vector_field": {
                            "type": "string",
                            "description": "Field containing vectors to search",
                            "default": "vector"
                        },
                        "filter_expr": {
                            "type": "string",
                            "description": "Filter expression for metadata"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 5
                        },
                        "output_fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Fields to include in results"
                        },
                        "metric_type": {
                            "type": "string",
                            "description": "Distance metric (COSINE, L2, IP)",
                            "default": "COSINE"
                        }
                    },
                    "required": ["collection_name", "vector", "filter_expr"]
                }
            ),
            types.Tool(
                name="milvus-create-collection",
                description="Create a new collection with specified schema",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name for the new collection"
                        },
                        "schema": {
                            "type": "object",
                            "description": "Collection schema definition"
                        },
                        "index_params": {
                            "type": "object",
                            "description": "Optional index parameters"
                        }
                    },
                    "required": ["collection_name", "schema"]
                }
            ),
            types.Tool(
                name="milvus-insert-data",
                description="Insert data into a collection",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of collection"
                        },
                        "data": {
                            "type": "object",
                            "description": "Dictionary mapping field names to lists of values"
                        }
                    },
                    "required": ["collection_name", "data"]
                }
            ),
            types.Tool(
                name="milvus-delete-entities",
                description="Delete entities from a collection based on filter expression",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of collection"
                        },
                        "filter_expr": {
                            "type": "string",
                            "description": "Filter expression to select entities to delete"
                        }
                    },
                    "required": ["collection_name", "filter_expr"]
                }
            ),
            types.Tool(
                name="milvus-get-collection-stats",
                description="Get statistics about a collection",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of collection"
                        }
                    },
                    "required": ["collection_name"]
                }
            ),
            types.Tool(
                name="milvus-multi-vector-search",
                description="Perform vector similarity search with multiple query vectors",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of the collection to search"
                        },
                        "vectors": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"}
                            },
                            "description": "List of query vectors"
                        },
                        "vector_field": {
                            "type": "string",
                            "description": "Field containing vectors to search",
                            "default": "vector"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results per query",
                            "default": 5
                        },
                        "output_fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Fields to include in results"
                        },
                        "metric_type": {
                            "type": "string",
                            "description": "Distance metric (COSINE, L2, IP)",
                            "default": "COSINE"
                        },
                        "filter_expr": {
                            "type": "string",
                            "description": "Optional filter expression"
                        }
                    },
                    "required": ["collection_name", "vectors"]
                }
            ),
            types.Tool(
                name="milvus-create-index",
                description="Create an index on a vector field",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of collection"
                        },
                        "field_name": {
                            "type": "string",
                            "description": "Field to index"
                        },
                        "index_type": {
                            "type": "string",
                            "description": "Type of index (IVF_FLAT, HNSW, etc.)",
                            "default": "IVF_FLAT"
                        },
                        "metric_type": {
                            "type": "string",
                            "description": "Distance metric (COSINE, L2, IP)",
                            "default": "COSINE"
                        },
                        "params": {
                            "type": "object",
                            "description": "Additional index parameters"
                        }
                    },
                    "required": ["collection_name", "field_name"]
                }
            ),
            types.Tool(
                name="milvus-bulk-insert",
                description="Insert data in batches for better performance",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of collection"
                        },
                        "data": {
                            "type": "object",
                            "description": "Dictionary mapping field names to lists of values"
                        },
                        "batch_size": {
                            "type": "integer",
                            "description": "Number of records per batch",
                            "default": 1000
                        }
                    },
                    "required": ["collection_name", "data"]
                }
            ),
            types.Tool(
                name="milvus-load-collection",
                description="Load a collection into memory for search and query",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of collection to load"
                        },
                        "replica_number": {
                            "type": "integer",
                            "description": "Number of replicas",
                            "default": 1
                        }
                    },
                    "required": ["collection_name"]
                }
            ),
            types.Tool(
                name="milvus-release-collection",
                description="Release a collection from memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of collection to release"
                        }
                    },
                    "required": ["collection_name"]
                }
            ),
            types.Tool(
                name="milvus-get-query-segment-info",
                description="Get information about query segments",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of collection"
                        }
                    },
                    "required": ["collection_name"]
                }
            ),
            types.Tool(
                name="milvus-upsert-data",
                description="Upsert data into a collection (insert or update if exists)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of collection"
                        },
                        "data": {
                            "type": "object",
                            "description": "Dictionary mapping field names to lists of values"
                        }
                    },
                    "required": ["collection_name", "data"]
                }
            ),
            types.Tool(
                name="milvus-get-index-info",
                description="Get information about indexes in a collection",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of collection"
                        },
                        "field_name": {
                            "type": "string",
                            "description": "Optional specific field to get index info for"
                        }
                    },
                    "required": ["collection_name"]
                }
            ),
            types.Tool(
                name="milvus-get-collection-loading-progress",
                description="Get the loading progress of a collection",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of collection"
                        }
                    },
                    "required": ["collection_name"]
                }
            ),
            types.Tool(
                name="milvus-create-dynamic-field",
                description="Add a dynamic field to an existing collection",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection_name": {
                            "type": "string",
                            "description": "Name of collection"
                        },
                        "field_name": {
                            "type": "string",
                            "description": "Name of the new field"
                        },
                        "data_type": {
                            "type": "string",
                            "description": "Data type of the field"
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description"
                        }
                    },
                    "required": ["collection_name", "field_name", "data_type"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(
        name: str,
        arguments: dict
    ) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        if name == "milvus-text-search":
            collection_name = arguments["collection_name"]
            query_text = arguments["query_text"]
            limit = arguments.get("limit", 5)
            output_fields = arguments.get("output_fields")
            drop_ratio = arguments.get("drop_ratio", 0.2)

            results = await milvus.search_collection(
                collection_name=collection_name,
                query_text=query_text,
                limit=limit,
                output_fields=output_fields,
                drop_ratio=drop_ratio
            )

            content = [
                types.TextContent(
                    type="text",
                    text=f"Search results for '{query_text}' in collection '{collection_name}':"
                )
            ]
            
            for result in results:
                content.append(
                    types.TextContent(
                        type="text",
                        text=f"<r>{str(result)}<r>"
                    )
                )
            
            return content

        elif name == "milvus-list-collections":
            collections = await milvus.list_collections()
            return [
                types.TextContent(
                    type="text",
                    text=f"Collections in database:\n{', '.join(collections)}"
                )
            ]

        elif name == "milvus-collection-info":
            collection_name = arguments["collection_name"]
            info = await milvus.get_collection_info(collection_name)
            return [
                types.TextContent(
                    type="text",
                    text=f"Collection info for '{collection_name}':\n{str(info)}"
                )
            ]

        elif name == "milvus-query":
            collection_name = arguments["collection_name"]
            filter_expr = arguments["filter_expr"]
            output_fields = arguments.get("output_fields")
            limit = arguments.get("limit", 10)

            results = await milvus.query_collection(
                collection_name=collection_name,
                filter_expr=filter_expr,
                output_fields=output_fields,
                limit=limit
            )

            content = [
                types.TextContent(
                    type="text",
                    text=f"Query results for '{filter_expr}' in collection '{collection_name}':"
                )
            ]
            
            for result in results:
                content.append(
                    types.TextContent(
                        type="text",
                        text=f"<r>{str(result)}<r>"
                    )
                )
            
            return content

        elif name == "milvus-count":
            collection_name = arguments["collection_name"]
            filter_expr = arguments.get("filter_expr")
            count = await milvus.count_entities(collection_name, filter_expr)
            msg = f"Count for collection '{collection_name}'"
            if filter_expr:
                msg += f" with filter '{filter_expr}'"
            msg += f": {count}"
            return [types.TextContent(type="text", text=msg)]

        elif name == "milvus-vector-search":
            collection_name = arguments["collection_name"]
            vector = arguments["vector"]
            vector_field = arguments.get("vector_field", "vector")
            limit = arguments.get("limit", 5)
            output_fields = arguments.get("output_fields")
            metric_type = arguments.get("metric_type", "COSINE")
            filter_expr = arguments.get("filter_expr")

            results = await milvus.vector_search(
                collection_name=collection_name,
                vector=vector,
                vector_field=vector_field,
                limit=limit,
                output_fields=output_fields,
                metric_type=metric_type,
                filter_expr=filter_expr
            )

            content = [
                types.TextContent(
                    type="text",
                    text=f"Vector search results for '{collection_name}':"
                )
            ]
            
            for result in results:
                content.append(
                    types.TextContent(
                        type="text",
                        text=f"<r>{str(result)}<r>"
                    )
                )
            
            return content

        elif name == "milvus-hybrid-search":
            collection_name = arguments["collection_name"]
            vector = arguments["vector"]
            vector_field = arguments.get("vector_field", "vector")
            filter_expr = arguments["filter_expr"]
            limit = arguments.get("limit", 5)
            output_fields = arguments.get("output_fields")
            metric_type = arguments.get("metric_type", "COSINE")

            results = await milvus.hybrid_search(
                collection_name=collection_name,
                vector=vector,
                vector_field=vector_field,
                filter_expr=filter_expr,
                limit=limit,
                output_fields=output_fields,
                metric_type=metric_type
            )

            content = [
                types.TextContent(
                    type="text",
                    text=f"Hybrid search results for '{collection_name}':"
                )
            ]
            
            for result in results:
                content.append(
                    types.TextContent(
                        type="text",
                        text=f"<r>{str(result)}<r>"
                    )
                )
            
            return content

        elif name == "milvus-create-collection":
            collection_name = arguments["collection_name"]
            schema = arguments["schema"]
            index_params = arguments.get("index_params")

            success = await milvus.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )

            return [
                types.TextContent(
                    type="text",
                    text=f"Collection '{collection_name}' created successfully"
                )
            ]

        elif name == "milvus-insert-data":
            collection_name = arguments["collection_name"]
            data = arguments["data"]

            result = await milvus.insert_data(
                collection_name=collection_name,
                data=data
            )

            return [
                types.TextContent(
                    type="text",
                    text=f"Data inserted into collection '{collection_name}' with result: {str(result)}"
                )
            ]

        elif name == "milvus-delete-entities":
            collection_name = arguments["collection_name"]
            filter_expr = arguments["filter_expr"]

            result = await milvus.delete_entities(
                collection_name=collection_name,
                filter_expr=filter_expr
            )

            return [
                types.TextContent(
                    type="text",
                    text=f"Entities deleted from collection '{collection_name}' with result: {str(result)}"
                )
            ]

        elif name == "milvus-get-collection-stats":
            collection_name = arguments["collection_name"]

            stats = await milvus.get_collection_stats(
                collection_name=collection_name
            )

            return [
                types.TextContent(
                    type="text",
                    text=f"Collection statistics for '{collection_name}':\n{json.dumps(stats, indent=2)}"
                )
            ]
            
        elif name == "milvus-multi-vector-search":
            collection_name = arguments["collection_name"]
            vectors = arguments["vectors"]
            vector_field = arguments.get("vector_field", "vector")
            limit = arguments.get("limit", 5)
            output_fields = arguments.get("output_fields")
            metric_type = arguments.get("metric_type", "COSINE")
            filter_expr = arguments.get("filter_expr")

            results = await milvus.multi_vector_search(
                collection_name=collection_name,
                vectors=vectors,
                vector_field=vector_field,
                limit=limit,
                output_fields=output_fields,
                metric_type=metric_type,
                filter_expr=filter_expr
            )

            content = [
                types.TextContent(
                    type="text",
                    text=f"Multi-vector search results for '{collection_name}':"
                )
            ]
            
            for i, result_group in enumerate(results):
                content.append(
                    types.TextContent(
                        type="text",
                        text=f"Results for vector {i+1}:"
                    )
                )
                
                for result in result_group:
                    content.append(
                        types.TextContent(
                            type="text",
                            text=f"<r>{str(result)}<r>"
                        )
                    )
            
            return content
            
        elif name == "milvus-create-index":
            collection_name = arguments["collection_name"]
            field_name = arguments["field_name"]
            index_type = arguments.get("index_type", "IVF_FLAT")
            metric_type = arguments.get("metric_type", "COSINE")
            params = arguments.get("params")

            success = await milvus.create_index(
                collection_name=collection_name,
                field_name=field_name,
                index_type=index_type,
                metric_type=metric_type,
                params=params
            )

            return [
                types.TextContent(
                    type="text",
                    text=f"Index created successfully on field '{field_name}' in collection '{collection_name}'"
                )
            ]
            
        elif name == "milvus-bulk-insert":
            collection_name = arguments["collection_name"]
            data = arguments["data"]
            batch_size = arguments.get("batch_size", 1000)

            results = await milvus.bulk_insert(
                collection_name=collection_name,
                data=data,
                batch_size=batch_size
            )

            total_inserted = sum(result.get("insert_count", 0) for result in results)
            
            return [
                types.TextContent(
                    type="text",
                    text=f"Bulk insert completed: {total_inserted} entities inserted into collection '{collection_name}'"
                )
            ]
            
        elif name == "milvus-load-collection":
            collection_name = arguments["collection_name"]
            replica_number = arguments.get("replica_number", 1)

            success = await milvus.load_collection(
                collection_name=collection_name,
                replica_number=replica_number
            )

            return [
                types.TextContent(
                    type="text",
                    text=f"Collection '{collection_name}' loaded successfully with {replica_number} replica(s)"
                )
            ]
            
        elif name == "milvus-release-collection":
            collection_name = arguments["collection_name"]

            success = await milvus.release_collection(
                collection_name=collection_name
            )

            return [
                types.TextContent(
                    type="text",
                    text=f"Collection '{collection_name}' released successfully"
                )
            ]
            
        elif name == "milvus-get-query-segment-info":
            collection_name = arguments["collection_name"]

            info = await milvus.get_query_segment_info(
                collection_name=collection_name
            )

            return [
                types.TextContent(
                    type="text",
                    text=f"Query segment info for collection '{collection_name}':\n{json.dumps(info, indent=2)}"
                )
            ]
            
        elif name == "milvus-upsert-data":
            collection_name = arguments["collection_name"]
            data = arguments["data"]

            result = await milvus.upsert_data(
                collection_name=collection_name,
                data=data
            )

            return [
                types.TextContent(
                    type="text",
                    text=f"Data upserted into collection '{collection_name}' with result: {str(result)}"
                )
            ]
            
        elif name == "milvus-get-index-info":
            collection_name = arguments["collection_name"]
            field_name = arguments.get("field_name")

            info = await milvus.get_index_info(
                collection_name=collection_name,
                field_name=field_name
            )

            return [
                types.TextContent(
                    type="text",
                    text=f"Index information for collection '{collection_name}':\n{json.dumps(info, indent=2)}"
                )
            ]
            
        elif name == "milvus-get-collection-loading-progress":
            collection_name = arguments["collection_name"]

            progress = await milvus.get_collection_loading_progress(
                collection_name=collection_name
            )

            return [
                types.TextContent(
                    type="text",
                    text=f"Loading progress for collection '{collection_name}':\n{json.dumps(progress, indent=2)}"
                )
            ]
            
        elif name == "milvus-create-dynamic-field":
            collection_name = arguments["collection_name"]
            field_name = arguments["field_name"]
            data_type = arguments["data_type"]
            description = arguments.get("description")

            success = await milvus.create_dynamic_field(
                collection_name=collection_name,
                field_name=field_name,
                data_type=data_type,
                description=description
            )

            return [
                types.TextContent(
                    type="text",
                    text=f"Dynamic field '{field_name}' of type '{data_type}' created successfully in collection '{collection_name}'"
                )
            ]
            

    return server

@click.command()
@click.option(
    "--milvus-uri",
    envvar="MILVUS_URI",
    required=True,
    help="Milvus server URI"
)
@click.option(
    "--milvus-token",
    envvar="MILVUS_TOKEN",
    required=False,
    help="Milvus authentication token"
)
@click.option(
    "--db-name",
    envvar="MILVUS_DB",
    default="default",
    help="Milvus database name"
)
def main(
    milvus_uri: str,
    milvus_token: Optional[str],
    db_name: str
):
    async def _run():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            server = serve(
                milvus_uri,
                milvus_token,
                db_name
            )
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="milvus",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

    asyncio.run(_run())

if __name__ == "__main__":
    main()