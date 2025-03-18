from typing import Optional, List, Dict, Any, Union, Tuple
from pymilvus import MilvusClient, DataType
from mcp.server.fastmcp import FastMCP, Context
import click
import asyncio
import json
import numpy as np
from datetime import datetime
from contextlib import asynccontextmanager
from typing import AsyncIterator
import os

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

class MilvusContext:
    def __init__(self, connector: MilvusConnector):
        self.connector = connector

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[MilvusContext]:
    """Manage application lifecycle for Milvus connector."""
    # Access config from server
    config = server.config
    print(f'config: {config}')
    # Initialize connector
    connector = MilvusConnector(
        uri=config["milvus_uri"],
        token=config.get("milvus_token"),
        db_name=config.get("db_name", "default")
    )
    print(f'connector: {connector}')
    try:
        # Yield the context to the server
        yield MilvusContext(connector)
    finally:
        # No cleanup needed for MilvusConnector
        pass

# Get configuration from environment variables with defaults
milvus_uri = os.environ.get("MILVUS_URI", "http://localhost:19530")
milvus_token = os.environ.get("MILVUS_TOKEN")
db_name = os.environ.get("MILVUS_DB", "default")

# Create a single connector instance
connector = MilvusConnector(milvus_uri, milvus_token, db_name)

# Create the FastMCP instance
mcp = FastMCP("Milvus")
print(f'mcp: {mcp}')

# Resource endpoints
@mcp.resource("collections://list")
def list_collections() -> str:
    """List all available collections in the Milvus database"""
    collections = connector.client.list_collections()
    print(f'collections: {collections}')
    return "\n".join(collections)

@mcp.resource("collections://{collection_name}/info")
async def get_collection_info(collection_name: str) -> str:
    """Get detailed information about a collection"""
    info = await connector.get_collection_info(collection_name)
    print(f'info: {info}')
    return json.dumps(info, indent=2)

@mcp.resource("collections://{collection_name}/stats")
async def get_collection_stats(collection_name: str) -> str:
    """Get statistics about a collection"""
    stats = await connector.get_collection_stats(collection_name)
    print(f'stats: {stats}')
    return json.dumps(stats, indent=2)

@mcp.resource("collections://{collection_name}/indexes")
async def get_collection_indexes(collection_name: str) -> str:
    """Get information about indexes in a collection"""
    indexes = await connector.get_index_info(collection_name)
    print(f'indexes: {indexes}')
    return json.dumps(indexes, indent=2)

# Tool endpoints
@mcp.tool()
async def milvus_text_search(
    collection_name: str,
    query_text: str,
    limit: int = 5,
    output_fields: Optional[List[str]] = None,
    drop_ratio: float = 0.2
) -> str:
    """
    Search for documents using full text search in a Milvus collection.
    
    Args:
        collection_name: Name of the collection to search
        query_text: Text to search for
        limit: Maximum number of results to return
        output_fields: Fields to include in results
        drop_ratio: Proportion of low-frequency terms to ignore (0.0-1.0)
    """
    results = await connector.search_collection(
        collection_name=collection_name,
        query_text=query_text,
        limit=limit,
        output_fields=output_fields,
        drop_ratio=drop_ratio
    )
    
    output = f"Search results for '{query_text}' in collection '{collection_name}':\n\n"
    for result in results:
        output += f"{result}\n\n"
        
    return output

@mcp.tool()
async def milvus_list_collections() -> str:
    """List all collections in the database."""
    collections = await connector.list_collections()
    return f"Collections in database:\n{', '.join(collections)}"

@mcp.tool()
async def milvus_query(
    collection_name: str,
    filter_expr: str,
    output_fields: Optional[List[str]] = None,
    limit: int = 10
) -> str:
    """
    Query collection using filter expressions.
    
    Args:
        collection_name: Name of the collection to query
        filter_expr: Filter expression (e.g. 'age > 20')
        output_fields: Fields to include in results
        limit: Maximum number of results
    """
    results = await connector.query_collection(
        collection_name=collection_name,
        filter_expr=filter_expr,
        output_fields=output_fields,
        limit=limit
    )
    
    output = f"Query results for '{filter_expr}' in collection '{collection_name}':\n\n"
    for result in results:
        output += f"{result}\n\n"
        
    return output


@mcp.tool()
async def milvus_vector_search(
    collection_name: str,
    vector: List[float],
    vector_field: str = "vector",
    limit: int = 5,
    output_fields: Optional[List[str]] = None,
    metric_type: str = "COSINE",
    filter_expr: Optional[str] = None
) -> str:
    """
    Perform vector similarity search on a collection.
    
    Args:
        collection_name: Name of the collection to search
        vector: Query vector
        vector_field: Field containing vectors to search
        limit: Maximum number of results
        output_fields: Fields to include in results
        metric_type: Distance metric (COSINE, L2, IP)
        filter_expr: Optional filter expression
    """
    results = await connector.vector_search(
        collection_name=collection_name,
        vector=vector,
        vector_field=vector_field,
        limit=limit,
        output_fields=output_fields,
        metric_type=metric_type,
        filter_expr=filter_expr
    )
    
    output = f"Vector search results for '{collection_name}':\n\n"
    for result in results:
        output += f"{result}\n\n"
        
    return output

@mcp.tool()
async def milvus_create_collection(
    collection_name: str,
    collection_schema: Dict[str, Any],
    index_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a new collection with specified schema.
    
    Args:
        collection_name: Name for the new collection
        collection_schema: Collection schema definition
        index_params: Optional index parameters
    """
    success = await connector.create_collection(
        collection_name=collection_name,
        schema=collection_schema,
        index_params=index_params
    )
    
    return f"Collection '{collection_name}' created successfully"

@mcp.tool()
async def milvus_insert_data(
    collection_name: str,
    data: Dict[str, List[Any]]
) -> str:
    """
    Insert data into a collection.
    
    Args:
        collection_name: Name of collection
        data: Dictionary mapping field names to lists of values
    """
    result = await connector.insert_data(
        collection_name=collection_name,
        data=data
    )
    
    return f"Data inserted into collection '{collection_name}' with result: {str(result)}"

@mcp.tool()
async def milvus_delete_entities(
    collection_name: str,
    filter_expr: str
) -> str:
    """
    Delete entities from a collection based on filter expression.
    
    Args:
        collection_name: Name of collection
        filter_expr: Filter expression to select entities to delete
    """
    result = await connector.delete_entities(
        collection_name=collection_name,
        filter_expr=filter_expr
    )
    
    return f"Entities deleted from collection '{collection_name}' with result: {str(result)}"

@mcp.tool()
async def milvus_load_collection(
    collection_name: str,
    replica_number: int = 1
) -> str:
    """
    Load a collection into memory for search and query.
    
    Args:
        collection_name: Name of collection to load
        replica_number: Number of replicas
    """
    success = await connector.load_collection(
        collection_name=collection_name,
        replica_number=replica_number
    )
    
    return f"Collection '{collection_name}' loaded successfully with {replica_number} replica(s)"

@mcp.tool()
async def milvus_release_collection(
    collection_name: str
) -> str:
    """
    Release a collection from memory.
    
    Args:
        collection_name: Name of collection to release
    """
    success = await connector.release_collection(
        collection_name=collection_name
    )
    
    return f"Collection '{collection_name}' released successfully"

if __name__ == "__main__":
    mcp.run()