#Load Raw PDF's

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

data_path = 'data/'
def pdf_load(data):
    loader = DirectoryLoader(data, 
                             glob= '*.pdf',
                             loader_cls= PyPDFLoader)
    docs = loader.load()
    return docs

documents = pdf_load(data=data_path)
print('Length of PDF Pages: ', len(documents))

#Create Chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,
                                                   chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print("Length of text docs:", len(text_chunks))

#Embed into vectors

def get_embedding_model():

    embedding_model= HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model= get_embedding_model()


#Store the embeddings in FAISS

FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks,embedding_model)

db.save_local(FAISS_PATH)