# RAG with plain text document using Langchain and OpenAI
# plain text file is from: The Project Gutenberg eBook of The Balkans: A History of Bulgaria—Serbia—Greece—Rumania—Turkey

!pip install langchain langchain-openai faiss-cpu langchain-community langchain-chroma

import os
os.environ["OPENAI_API_KEY"] = 'your_api_key'

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

print(text[:50])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(text)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

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
