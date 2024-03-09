import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

import dotenv
dotenv.load_dotenv()

# api keys
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]


def get_pinecone_client():
    return Pinecone(api_key=PINECONE_API_KEY)


def get_vectorstore(index_name: str, namespace: str, emb_model: str = "text-embedding-3-small") -> PineconeVectorStore:
    """Returns a Pinecone DB connection with openai embeddings"""
    pc = Pinecone(api_key=PINECONE_API_KEY)               # TODO: do this one time outside and pass inside everything
    if index_name not in pc.list_indexes().names():
        print(f"Index {index_name} does not exist")
        create_index(pc, index_name)

    embedding_model = OpenAIEmbeddings(
        model=emb_model,
        openai_api_key=OPENAI_API_KEY
    )

    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding_model,
        namespace=namespace
    )


def create_index(pc: Pinecone, index: str):
    """Create a new index in Pinecone"""
    if index not in pc.list_indexes().names():
        print(f"Creating index {index}...")
        pc.create_index(
            name=index,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-west-2"
            )
        )
        # wait for index to be initialized
        while not pc.describe_index(index).status['ready']:
            time.sleep(1)
    else:
        print(f"Index {index} already exists")


def delete_index(pc: Pinecone, index: str):
    """Delete existing index in Pinecone"""
    if index in pc.list_indexes().names():
        print(f"Deleting index {index}...")
        pc.delete_index(index)
    else:
        print(f"Index {index} does not exist -- cannot delete")


def save_embeddings(pc: Pinecone, embeddings: list, site_id: str, index_name: str):
    """Store the embeddings for the site in Pinecone"""
    index = pc.Index(index_name)
    index.upsert(embeddings, namespace=site_id)


if __name__ == "__main__":
    ...

