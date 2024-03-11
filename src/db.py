import time
from .embedder import Embedder
from pinecone import Pinecone, ServerlessSpec
from typing import Union


class PineconeDB():
    """
    Class to handle Pinecone DB connections and operations with cached indexes and namespaces
    """
    def __init__(
            self, 
            client: Pinecone = None,
        ) -> None:
        """
        Create a new PineconeDB connection
        
        Args:
            client: Pinecone client
        """
        if client is None:
            raise ValueError("Pinecone client is required")
        self.client: Pinecone = client      

        # create cache to minimize API calls: map index names to Pinecone.Index objects + list of namespaces
        # each index has format: {"index_name": {"index": Pinecone.Index, "namespaces": [namespace1, namespace2, ...]}}
        self.indexes: dict[str, dict[str, Union[Pinecone.Index, list[str]]]] = {}


    def __str__(self) -> str:
        index_info = []
        for index_name, index_data in self.indexes.items():
            namespace_count = len(index_data["namespaces"])
            index_info.append(f"{index_name} (namespaces: {namespace_count})")
        
        index_str = ", ".join(index_info)
        return f"PineconeDB(indexes=[{index_str}])"


    def get_index(self, index_name: str, embedder: Embedder) -> Pinecone.Index:
        """Get or create an index object"""
        if index_name not in self.indexes:
            # if not cached, check if it exists in the database
            if index_name not in self.client.list_indexes().names():
                # if not in the database, create it
                self.create_index(index_name, embedder)

            # create a new index object and add it to the cache
            index = self.client.Index(index_name)
            self.indexes[index_name] = {
                "index": index,
                "namespaces": []
            }
        return self.indexes[index_name]["index"]


    def create_index(self, index_name: str, embedder: Embedder) -> None:
        """Create a new index in Pinecone"""
        if index_name not in self.client.list_indexes().names():
            print(f"Creating index {index_name}...")
            try:
                self.client.create_index(
                    name=index_name,
                    dimension=embedder.dim,
                    metric=embedder.metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-west-2"
                    )
                )
                # wait for index to be initialize\=_?? 
                -+
                while not self.client.describe_index(index_name).status['ready']:
                    time.sleep(1)

                # add index to cache
                self.indexes[index_name] = {
                    "index": self.client.Index(index_name),
                    "namespaces": []
                }
            except Exception as e:
                print(f"Error creating index {index_name}: {str(e)}")
                raise
        else:
            print(f"Index {index_name} already exists")


    def delete_index(self, index_name: str) -> None:
        """Delete an existing index in Pinecone"""
        # check that index is cached first before making api call to check if it exists
        if index_name in self.indexes or index_name in self.client.list_indexes().names():
            print(f"Deleting index {index_name}...")
            try:
                # remove index from cache if present
                if index_name in self.indexes:
                    del self.indexes[index_name]
                # delete index from db
                self.client.delete_index(index_name)
            except Exception as e:
                print(f"Error deleting index {index_name}: {str(e)}")
                raise
        else:
            print(f"Index {index_name} does not exist -- cannot delete")






    def save_embeddings(self, index_name: str, namespace: str, embedding: list[float]) -> None:
        """Store an embedding in Pinecone"""
        index = self.client.Index(index_name)
        index.upsert([embedding], namespace=namespace)


    def save_batch_embeddings(self, index_name: str, namespace: str, embeddings: list[list[float]]) -> None:
        """Store a batch of embeddings in Pinecone"""
        index = self.client.Index(index_name)
        index.upsert(embeddings, namespace=namespace)

    
    def format_query(self, query: str) -> str:
        pass
        '''
        If a tuple is used, it must be of the form (id, values, metadata) or (id, values). where id is a string, vector is a list of floats, metadata is a dict,
        and sparse_values is a dict of the form {'indices': List[int], 'values': List[float]}.
        '''

    # https://github.com/langchain-ai/langchain/blob/master/libs/partners/pinecone/langchain_pinecone/vectorstores.py
    # https://docs.pinecone.io/reference/describe_index_stats
    # https://docs.pinecone.io/reference/describe_index
    def query(self, index_name: str, namespace: str, query: str, embedder: Embedder, top_k: int = 5) -> list[dict[str, str]]:
        """Query Pinecone for similar embeddings"""
        if index_name not in self.client.list_indexes().names():
            raise ValueError(f"Index {index_name} does not exist")
        
        index_description = self.client.describe_index(index_name)
        if embedder.dim != index_description.dimension:
            raise ValueError(f"Embedder dimension {embedder.dim} does not match index dimension {index_description.dimension}")
        if embedder.metric != index_description.metric:
            raise ValueError(f"Embedder metric {embedder.metric} does not match index metric {index_description.metric}")

        index = self.client.Index(index_name)
        index_stats_response = index.describe_index_stats()

        if namespace not in index_stats_response["namespaces"].keys():
            raise ValueError(f"Namespace {namespace} does not exist in index {index_name}")
        
        # if namespace not in self.client.describe_index(index_name).namespaces:
        #     raise ValueError(f"Namespace {namespace} does not exist in index {index_name}")
        return index.query(
            namespace=namespace,
            vector=embedder.embed_text(query),
            top_k=top_k
        )["matches"]

    def query(self, index_name: str, namespace: str, query: str, embedder: Embedder, top_k: int = 5) -> list[dict[str, str]]:
        """Query Pinecone for similar embeddings"""
        try:
            index = self.client.Index(index_name)
            index_description = self.client.describe_index(index_name)
            index_stats = index.describe_index_stats()
        except Exception as e:
            raise ValueError(f"Error retrieving index information: {str(e)}")

        if embedder.dim != index_description.dimension:
            raise ValueError(f"Embedder dimension {embedder.dim} does not match index dimension {index_description.dimension}")
        if embedder.metric != index_description.metric:
            raise ValueError(f"Embedder metric {embedder.metric} does not match index metric {index_description.metric}")
        if namespace not in index_stats["namespaces"].keys():
            raise ValueError(f"Namespace {namespace} does not exist in index {index_name}")

        return index.query(
            namespace=namespace,
            vector=embedder.embed_text(query),
            top_k=top_k
        )["matches"]

    def query_batch(self, index_name: str, namespace: str, queries: list[str], embedder: Embedder, top_k: int = 5) -> list[list[str]]:
        """Query Pinecone for similar embeddings"""
        if index_name not in self.client.list_indexes().names():
            raise ValueError(f"Index {index_name} does not exist")
        if namespace not in self.client.describe_index(index_name).namespaces:
            raise ValueError(f"Namespace {namespace} does not exist in index {index_name}")
        index = self.client.Index(index_name)
        queries = [embedder.embed_text(query) for query in queries]
        return index.query(queries=queries, top_k=top_k, namespace=namespace).ids