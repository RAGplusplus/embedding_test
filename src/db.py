import time
from .embedder import Embedder
from .document import Document
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
    
    
    def __repr__(self) -> str:
        return self.__str__()


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
                # wait for index to be initialized
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


    def format_doc(self, doc: Document, embedder: Embedder) -> dict[str, str]:
        """Format a document as a Pinecone entry"""
        embedding: list[float] = embedder.embed_text(doc.text)
        return doc.to_dict(embedding)
    

    def format_batch_docs(self, docs: list[Document], embedder: Embedder) -> list[dict[str, str]]:
        """Format a batch of documents as Pinecone entries"""
        embeddings: list[list[float]] = embedder.embed_batch([doc.text for doc in docs])
        return [self.format_data(doc, embedding) for doc, embedding in zip(docs, embeddings)]


    def upsert_doc(self, index_name: str, namespace: str, embedder: Embedder, doc: Document) -> None:
        """Store an embedding in Pinecone"""
        try:
            index: Pinecone.Index = self.get_index(index_name)
            data: dict[str, str] = self.format_doc(doc, embedder)
            index.upsert([data], namespace=namespace)
        except Exception as e:
            print(f"Error upserting document to index {index_name} in namespace {namespace}: {str(e)}")
            raise


    def upsert_batch(self, index_name: str, namespace: str, embedder: Embedder, docs: list[Document]) -> None:
        """Store a batch of embeddings in Pinecone"""
        try:
            index: Pinecone.Index = self.get_index(index_name)
            data: list[dict[str, str]] = self.format_batch_docs(docs, embedder)
            index.upsert(data, namespace=namespace)
        except Exception as e:
            print(f"Error upserting batch of documents to index {index_name} in namespace {namespace}: {str(e)}")
            raise


    # https://github.com/langchain-ai/langchain/blob/master/libs/partners/pinecone/langchain_pinecone/vectorstores.py
    # https://docs.pinecone.io/reference/describe_index_stats
    # https://docs.pinecone.io/reference/describe_index
    def query(self, index_name: str, namespace: str, query: str, embedder: Embedder, top_k: int = 5) -> list[Document]:
        """Query Pinecone for similar embeddings. Returns list of top_k Documents in the database."""
        try:
            index = self.get_index(index_name)
            index_description = self.client.describe_index(index_name)
            index_stats = index.describe_index_stats()
        except Exception as e:
            raise ValueError(f"Error retrieving index information: {str(e)}")

        # ensure that the embedder is compatible with the index and that the namespace exists
        if embedder.dim != index_description.dimension:
            raise ValueError(f"Embedder dimension {embedder.dim} does not match index dimension {index_description.dimension}")
        if embedder.metric != index_description.metric:
            raise ValueError(f"Embedder metric {embedder.metric} does not match index metric {index_description.metric}")
        if namespace not in index_stats["namespaces"].keys():
            raise ValueError(f"Namespace {namespace} does not exist in index {index_name}")

        # get the top_k matches
        matches = index.query(
            namespace=namespace,
            vector=embedder.embed_text(query),
            top_k=top_k
        )["matches"]

        # format the matches as Document objects
        return [
            Document(
                source = match["metadata"]["source"],
                text = match["metadata"]["text"],
                id = match["id"]
            )
            for match in matches
        ]
    

    def query_batch(self, index_name: str, namespace: str, queries: list[str], embedder: Embedder, top_k: int = 5) -> list[list[Document]]:
        """Query Pinecone for similar embeddings. Returns a list of lists of top_k Documents for each query."""
        try:
            index = self.get_index(index_name)
            index_description = self.client.describe_index(index_name)
            index_stats = index.describe_index_stats()
        except Exception as e:
            raise ValueError(f"Error retrieving index information: {str(e)}")

        # ensure that the embedder is compatible with the index and that the namespace exists
        if embedder.dim != index_description.dimension:
            raise ValueError(f"Embedder dimension {embedder.dim} does not match index dimension {index_description.dimension}")
        if embedder.metric != index_description.metric:
            raise ValueError(f"Embedder metric {embedder.metric} does not match index metric {index_description.metric}")
        if namespace not in index_stats["namespaces"].keys():
            raise ValueError(f"Namespace {namespace} does not exist in index {index_name}")

        # get the top_k matches for each query
        matches_list = index.query(
            namespace=namespace,
            queries=embedder.embed_batch(queries),
            top_k=top_k
        )

        # format the matches as Document objects
        return [
            [
                Document(
                    source=match["metadata"]["source"],
                    text=match["metadata"]["text"],
                    id=match["id"]
                )
                for match in matches
            ]
            for matches in matches_list
        ]