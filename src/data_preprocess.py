"""
File to convert CISI data to Pinecone format


CISI.ALL:
A file of 1,460 "documents" each with a unique ID (.I), title (.T), author (.A), abstract (.W) and list of cross-references to other documents (.X). It is the dataset for training IR models when used in conjunction with the Queries (CISI.QRY).

CISI.QRY:
A file containing 112 queries each with a unique ID (.I) and query text (.W).

CISI.REL:
A file containing the mapping of query ID (column 0) to document ID (column 1). A query may map to more than one document ID. This file contains the "ground truth" that links queries to documents. Use this to train and test your algorithm.

"""

def Document():
    """Class to represent a document in the CISI dataset"""
    def __init__(self, id, title, author, abstract, cross_references):
        self.id = id
        self.title = title
        self.author = author
        self.abstract = abstract
        self.cross_references = cross_references

def Query():
    """Class to represent a query in the CISI dataset"""
    def __init__(self, id, query_text):
        self.id = id
        self.query_text = query_text

def Relationship():
    """Class to represent a relationship between a query and a document in the CISI dataset"""
    def __init__(self, query_id, document_id):
        self.query_id = query_id
        self.document_id = document_id