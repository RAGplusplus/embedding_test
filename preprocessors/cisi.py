"""
File to preprocess CISI data and store it as json files. The CISI dataset is a collection of documents,
queries, and relationships between queries and documents. The documents are stored in the file CISI.ALL,
the queries are stored in the file CISI.QRY, and the relationships are stored in the file CISI.REL.

Dataset: https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval/data

File information:

    CISI.ALL:
    A file of 1,460 "documents" each with a unique ID (.I), title (.T), author (.A), abstract (.W) and list 
    of cross-references to other documents (.X). It is the dataset for training IR models when used in 
    conjunction with the Queries (CISI.QRY).

    CISI.QRY:
    A file containing 112 queries each with a unique ID (.I) and query text (.W).

    CISI.REL:
    A file containing the mapping of query ID (column 0) to document ID (column 1). A query may map to 
    more than one document ID. This file contains the "ground truth" that links queries to documents. Use 
    this to train and test your algorithm.

Code for functions read_documents(), read_queries(), and read_mappings() is modified from: https://www.kaggle.com/code/aleksandrmorozov123/nlp-with-python
"""
from ..src.io import save_json
import multiprocessing
import os



'''
data/cisi/raw
data/cisi/processed
data/cisi/processed/documents.json
[
    {
        "id": id,
        "values": None,
        "metadata": {
            "source": source,
            "text": text
        }
    }
    {
        "id": id,
        "values": None,
        "metadata": {
            "source": source,
            "text": text
        }
    }
]
data/cisi/processed/queries.json
[
    {
        "query": query,
    }

]
'''














class Document():
    """Class to represent a document in the CISI dataset"""
    def __init__(self, id, title, author, abstract, cross_references):
        self.id = id
        self.title = title
        self.author = author
        self.abstract = abstract
        self.cross_references = cross_references

class Query():
    """Class to represent a query in the CISI dataset"""
    def __init__(self, id, query_text):
        self.id = id
        self.query_text = query_text

class Relationship():
    """Class to represent a relationship between a query and a document in the CISI dataset"""
    def __init__(self, query_id, document_id):
        self.query_id = query_id
        self.document_id = document_id


def read_documents ():
    f = open ("/kaggle/input/cisi-a-dataset-for-information-retrieval/CISI.ALL")
    merged = " "
    # the string variable merged keeps the result of merging the field identifier with its content
    
    for a_line in f.readlines ():
        if a_line.startswith ("."):
            merged += "\n" + a_line.strip ()
        else:
            merged += " " + a_line.strip ()
    # updates the merged variable using a for-loop
    
    documents = {}
    
    content = ""
    doc_id = ""
    # each entry in the dictioanry contains key = doc_id and value = content
    
    for a_line in merged.split ("\n"):
        if a_line.startswith (".I"):
            doc_id = a_line.split (" ") [1].strip()
        elif a_line.startswith (".X"):
            documents[doc_id] = content
            content = ""
            doc_id = ""
        else:
            content += a_line.strip ()[3:] + " "
    f.close ()
    return documents

# print out the size of the dictionary and the content of the very first article
documents = read_documents ()
print (len (documents))
print (documents.get ("1"))


def read_queries ():
    f = open ("/kaggle/input/cisi-a-dataset-for-information-retrieval/CISI.QRY")
    merged = ""
    
    # merge the conten of each field with its identifier and separate different fields with lune breaks
    for a_line in f.readlines ():
        if a_line.startswith ("."):
            merged += "\n" + a_line.strip ()
        else:
            merged += " " + a_line.strip ()
    
    queries = {}
    
    # initialize queries dictionary with key = qry_id and value=content for each query in the dataset
    content = ""
    qry_id = ""
    
    for a_line in merged.split ("\n"):
        if a_line.startswith (".I"):
            if not content == "":
                queries [qry_id] = content
                content = ""
                qry_id = ""
            # add an enrty to the dictionary when you encounter an .I identifier
            qry_id = a_line.split(" ")[1].strip ()
        # otherwise, keep adding content to the content variable
        elif a_line.startswith (".W") or a_line.startswith (".T"):
            content += a_line.strip ()[3:] + " "
    queries [qry_id] = content
    f.close ()
    return queries

# print out the length of the dictionary and the content of the first query
queries = read_queries ()
print (len (queries))
print (queries.get("1"))

def read_mappings ():
    f = open ("/kaggle/input/cisi-a-dataset-for-information-retrieval/CISI.REL")
    mappings = {}
    
    for a_line in f.readlines ():
        voc = a_line.strip ().split ()
        key = voc[0].strip ()
        current_value = voc[1].strip()
        value = []
        # update the entry in the mappings dictionary with the current value
        if key in mappings.keys ():
            value = mappings.get (key)
        value.append (current_value)
        mappings [key] = value
    f.close ()
    return mappings

# print out some information about the mapping data structure
mappings = read_mappings ()
print (len (mappings))
print (mappings.keys ())
print (mappings.get ("1"))