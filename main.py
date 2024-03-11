import os
import dotenv
from colorama import Fore
from openai import OpenAI
from pinecone import Pinecone
from utils.db import PineconeDB

# load environment variables
dotenv.load_dotenv()

# check if all required keys are set
required_keys = [
    "OPENAI_API_KEY", 
    "PINECONE_API_KEY", 
    "LANGCHAIN_API_KEY"
]

for key in required_keys:
    val = os.environ.get(key, None)
    if val is None:
        print(f"{Fore.RED}Error: {key} is not set in .env file")
        exit(1)
    if val is "":
        print(f"{Fore.RED}Error: {key} is empty in .env file")
        exit(1)

# get the keys and create clients
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]   # TODO: where is this actually used???

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# create a Pinecone connection 
pc = PineconeDB(pinecone_client)

if __name__ == "__main__":
    ...