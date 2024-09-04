# Simple RAG SQL generation with relational database
# The database can be downloaded from : https://www.kaggle.com/datasets/johnp47/sql-murder-mystery-database

!pip install --upgrade --quiet  langchain langchain-community langchain-openai 

import os
os.environ['OPENAI_API_KEY']='your_key'

import sqlite3
conn = sqlite3.connect('sql-murder-mystery.db')
cursor = conn.cursor()

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///sql-murder-mystery.db")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM crime_scene_report LIMIT 10;")

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query
chain.invoke({"question": "How many murders are there?"})

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

chain.invoke({"question": "How many blackmails are in Albuquerque?"})

from langchain_community.agent_toolkits import create_sql_agent

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

agent_executor.invoke(
    {
        "input": "List the persons wiht drivers licence who said nothing on the interview?"
    }
)

agent_executor.invoke({"input": "Describe the interview table"})
