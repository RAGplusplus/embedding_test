from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    ChararacterTextSplitter,


)

class Chunker():
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size


    def chunk(self, data):
        return [data[i:i+self.chunk_size] for i in range(0, len(data), self.chunk_size)]