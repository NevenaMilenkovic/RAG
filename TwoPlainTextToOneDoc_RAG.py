# Simple RAG that uses 2 Plain text documents, the contents of documents are added to one document before invoking the model
# The documents used are: The Project Gutenberg eBook of The Balkans: A History of Bulgaria—Serbia—Greece—Rumania—Turkey and The Project Gutenberg eBook of The Frontiers of Language and Nationality in Europe

!pip install langchain langchain-openai faiss-cpu langchain-community langchain-chroma

import os
os.environ['OPENAI_API_KEY']='your_api_key'

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_community.document_loaders import TextLoader
loader = TextLoader(file_path='/content/pg11716.txt')
text = loader.load()

loader2 = TextLoader(file_path='/content/pg58205.txt')
text2 = loader2.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits1 = text_splitter.split_documents(text)
splits2 = text_splitter.split_documents(text2)

documents = splits1 + splits2

vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("Why did Turks occupy Balkans?")

rag_chain.invoke("What dialect is spoken between Bulgaria and Greece?")
