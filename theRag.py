from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma # Our ChormaDB
from langchain_core.documents import Document # Pass in documents to our ChromaDB
import os
import pandas as pd # Data Frame to read in our CSV file
# IMPORTING ALL THE RELEVANT LIBRARIES AND PACKAGES FOR OUR LOCAL LLM AND CHROMADB

dataFrame = pd.read_csv("philippine_agriculture_knowledge_guides.csv") #subject to change

embeddings = OllamaEmbeddings(model="mxbai-embed-large") # Our Embeddings model

db_location = "./chroma_db" # Location to store our ChromaDB >> FOLDER
add_document = not os.path.exists(db_location) # Check if the ChromaDB exists

if add_document:
    documents = [] # the list of the  documents
    ids = [] # store the id's and separate the lists(documents), also to identify the documents in the ChromaDB

    for i, row in dataFrame.iterrows(): # iterate through the Data Frame / CSV file
        document = Document(
            #id,category,title,content,commodity,location_scope,agency,season,keywords,language,last_reviewed

            page_content=row["title"] + " " + row["content"], # subject to change
            metadata={"category": row["category"],"last_reviewed": row["last_reviewed"]},
            id=str(i) # id is the index of the row in the Data Frame
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="philippine_agriculture_knowledge_guides",
    persist_directory=db_location, # store it than becoming a memory, avoid generating the embeddings again, and save time
    embedding_function=embeddings   
)

if add_document:
    vector_store.add_documents(documents=documents, ids=ids) # add the documents to the ChromaDB

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5} # retrieve the top 5 relevant documents from the ChromaDB, subject to change
) 