import uuid

class Document():
    """Class to hold data in Pinecone's Document format"""

    def __init__(self, source: str, text: str, id: str = None) -> None:
        self.source: str = source
        self.text: str = text
        self.id: str = str(uuid.uuid4()) if None else id

    def __str__(self) -> str:  
        return f"Document(id={self.id}, source={self.source}, text={self.text[:50]}...)"

    def __repr__(self) -> str:
        return self.__str__()
    
    '''
    If a tuple is used, it must be of the form (id, values, metadata) or (id, values). where id is a string, vector is a list of floats, metadata is a dict,
    and sparse_values is a dict of the form {'indices': List[int], 'values': List[float]}.
    '''
    def to_dict(self, embedding: list[float] = None) -> dict[str, any]:
        """Return Pinecone's format as a dictionary"""
        return {
            "id": self.id,
            "values": embedding,
            "metadata": {
                "source": self.source,
                "text": self.text
            }
        }
