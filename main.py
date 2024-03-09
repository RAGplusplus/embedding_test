import os
import dotenv
from colorama import Fore
# from utils.db import connect

dotenv.load_dotenv()

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
    

if __name__ == "__main__":
    ...