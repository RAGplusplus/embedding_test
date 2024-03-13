class DataPreprocessor():
    def __init__(self, config):
        self.config = config
        self.documents = self.read_documents()
        self.queries = self.read_queries()
        self.relevance = self.read_relevance()