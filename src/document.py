import uuid

class Document():
    """Class to hold data in Pinecone's Document format"""

    def __init__(self, source: str, text: str) -> None:
        self.id: str = str(uuid.uuid4())
        self.source: str = source
        self.text: str = text

    def __str__(self) -> str:  
        return f"Document(id={self.id}, source={self.source}, text={self.text[:50]}...)"

    def __repr__(self) -> str:
        return self.__str__()
    
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