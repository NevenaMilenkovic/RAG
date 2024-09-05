# RAG with input of multiple plain text documents manualy categoriezed using multiple vectorstores
# Plain text documents are: The Project Gutenberg eBook of Famous Men of the Middle Ages, The Project Gutenberg eBook of Napoleon Bonaparte, The Project Gutenberg eBook of The Balkans: A History of Bulgaria—Serbia—Greece—Rumania—Turkey,
# The Project Gutenberg eBook of The Balkan Wars: 1912-1913, The Project Gutenberg eBook of Servian Popular Poetry, The Project Gutenberg eBook of The Frontiers of Language and Nationality in Europe,
# The Project Gutenberg eBook of Europe Since 1918, The Project Gutenberg eBook of Wars & Treaties, 1815 to 1914, The Project Gutenberg eBook of Serbian Fairy Tales, The Project Gutenberg eBook of Short stories from the Balkans

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

from langchain_core.documents import Document
def load_documents_with_categories(file_paths, categories):
    documents = []
    for file_path, category in zip(file_paths, categories):
        with open(file_path, 'r') as f:
            content = f.read()
        document = Document(page_content=content, metadata={"category": category})
        documents.append(document)
    return documents

file_paths = ["/content/pg11716.txt", "/content/pg58205.txt","/content/pg67191.txt","/content/pg39028.txt",
              "/content/pg3725.txt","/content/pg73663.txt","/content/pg36192.txt","/content/pg59573.txt","/content/pg60026.txt","/content/pg3775.txt"]
categories = ["Balkan History", "Language Europe","Serbian Fairy Tales","Serbian popular poetry",
              "Famous Middle Ages History","Balkan Short Stories","Balkan wars","Europe history after 1918", "Wars Peace","Napoleon"]

documents = load_documents_with_categories(file_paths, categories)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

vectorstores = {}
for category in categories:
    category_splits = [split for split in splits if split.metadata["category"] == category]
    vectorstore = Chroma.from_documents(documents=category_splits, embedding=OpenAIEmbeddings())
    vectorstores[category] = vectorstore

retrievers = {category: vectorstore.as_retriever() for category, vectorstore in vectorstores.items()}

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def select_retriever(query):
    for category, retriever in retrievers.items():
        if category in query:
            return retriever
    return retrievers.get("default_category", retrievers[list(retrievers.keys())[0]])

def create_rag_chain(query):
    return (
        {"context": select_retriever(query) | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

query1="How long was Turkish occupation of Bulgaria?"
rag_chain1 = create_rag_chain(query1)
response1 = rag_chain1.invoke(query1)
print(response1)

query2="What language is spoken in Bulgaria by minority?"
rag_chain2 = create_rag_chain(query2)
response2 = rag_chain2.invoke(query2)
print(response2)

query3="Did the old couple stay under the pine-tree?"
rag_chain3 = create_rag_chain(query3)
response3 = rag_chain3.invoke(query3)
print(response3)

query4="What was the name of Jelitza youngest brother?"
rag_chain4 = create_rag_chain(query4)
response4 = rag_chain4.invoke(query4)
print(response4)

query5="Who was succesor of King Connard?"
rag_chain5 = create_rag_chain(query5)
response5 = rag_chai5.invoke(query5)
print(response5)

query6="Was Jona fearfull?"
rag_chain6 = create_rag_chain(query6)
response6 = rag_chain6.invoke(query6)
print(response6)

query7="Was Serbia led by Kara-george?"
rag_chain7 = create_rag_chain(query7)
response7 = rag_chain7.invoke(query7)
print(response7)

query8="Why did France sponsor Poles?"
rag_chain8 = create_rag_chain(query8)
response8 = rag_chain8.invoke(query8)
print(response8)

query9="Was Lorraine part of Germany state?"
rag_chain9 = create_rag_chain(query9)
response9 = rag_chain2.invoke(query9)
print(response9)

query10="Which nation Napoleon wanted Italians to be friendly with?"
rag_chain10 = create_rag_chain(query10)
response10 = rag_chain10.invoke(query10)
print(response10)
