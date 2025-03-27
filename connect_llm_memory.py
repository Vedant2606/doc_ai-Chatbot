import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

#Setup LLM
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
#Langsmith Tracing

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.5,
    model_kwargs={"top_p": 0.9, "frequency_penalty": 0.2}
)

#Connect LLM with FAISS and create chain

custom_prompt_template= """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template= custom_prompt_template, input_variables=['context', 'question'])
    return prompt


FAISS_PATH  = "vectorstore/db_faiss"
embedding_model= HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db =  FAISS.load_local(FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

#Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm= llm, 
    chain_type = 'stuff',
    retriever= db.as_retriever(search_kwargs = {'k':3}),
    return_source_documents= True,
    chain_type_kwargs = {'prompt': set_custom_prompt(custom_prompt_template)}
    )

#Invoke with single Query

user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT: ", response['result'])
print("SOURCE DOCUMENTS: ", response['source_documents'])