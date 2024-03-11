import tiktoken
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SemanticChunker
)
from typing import Callable, Union


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_string(string: str, encoding_name: str = "CL100K_Base") -> int:
    """Returns the number of tokens in a text string: CL100K_Base is encoding for gpt models"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# https://python.langchain.com/docs/modules/data_connection/document_transformers/
strategies: dict[str, Union[RecursiveCharacterTextSplitter, CharacterTextSplitter]]= {
    "character": CharacterTextSplitter,           # split by single characters
    "recursive": RecursiveCharacterTextSplitter,  # split list of chars to keep paragraphs, sentences, and then words together for as long as possible
    "semantic":  SemanticChunker,                 # splits into sentences, then groups into groups of 3 sentences, and then merges ones that are similar in the embedding space
}


# *** callable takes a string and returns an int ***
length_functions: dict[str, Callable[[str], int]] = {
    "char": len,
    "token": num_tokens_from_string
}


class Chunker():
    """
    Class to handle chunking of text
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "recursive",
        length_function: str = "char",
        # keep_separator: bool = False,  # TODO: prob remove
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """
        Create a new Chunker

        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            strategy: Strategy to use for splitting text
            length_function: Function that measures the length of given chunks (char or token)
            keep_separator: Whether to keep the separator in the chunks
            add_start_index: If `True`, includes chunk's start index in metadata
            strip_whitespace: If `True`, strips whitespace from the start and end of
                              every document
        """
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.strategy: str = strategy.lower()
        self.length_function: str = length_function.lower()
        # self.keep_separator: bool = keep_separator
        self.add_start_index: bool = add_start_index    
        self.strip_whitespace: bool = strip_whitespace

        # get text splitter
        if self.strategy in strategies:
            if self.strategy == "semantic":
                self.splitter = strategies[self.strategy](
                            chunk_size=self.chunk_size,
                            chunk_overlap=self.overlap,
                            length_function=length_functions[self.length_function],
                            # normalize_function=normalize_l2,
                            # api_key=api_key
                        )
            else:
                self.splitter = strategies[self.strategy](
                                chunk_size=self.chunk_size,
                                chunk_overlap=self.overlap,
                                length_function=length_functions[self.length_function],
                            )
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}. Valid strategies are: {", ".joinlist(strategies.keys())}")
        

        self.splitter =  CharacterTextSplitter(
                            separator="\n\n",
                            chunk_size=self.chunk_size,
                            chunk_overlap=self.overlap,
                            length_function=length_functions[self.length_function],
                            is_separator_regex=False,
                        )

    def split(self, text):
        '''
        metadatas = [{"document": 1}, {"document": 2}]
        documents = text_splitter.create_documents(
            [state_of_the_union, state_of_the_union], metadatas=metadatas
        )
        print(documents[0])
        
        '''
        return self.splitter.create_documents([text])
    

    def split_code(self, text, lang: str = "py"):
        ...

    def chunk(self, data):
        return [data[i:i+self.chunk_size] for i in range(0, len(data), self.chunk_size)]