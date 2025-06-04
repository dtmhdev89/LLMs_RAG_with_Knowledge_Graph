import os
import datetime
from typing import Literal, Optional, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


class SubQuery(BaseModel):
    """Decompose a given question/query into sub-queries"""

    sub_query: str = Field(
        ...,
        description="A unique paraphrasing of the original questions.",
    )


class Decompose:
    def __init__(
        self,
        provider: Literal['ollama', 'openai'],
        model_name: str
    ):
        if provider == 'ollama':
            self.llm = ChatOllama(
                model=model_name,
                temperature=0,
                base_url=os.getenv("OLLAMA_CONNECT_URL")
            )
        elif provider == 'openai':
            self.llm = ChatOpenAI(
                model=model_name, 
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
    @property
    def system_prompt(self):
        """system prompt for chain"""

        system = """You are an expert at converting user questions into Neo4j Cypher queries. \

        Perform query decomposition. Given a user question, break it down into two distinct subqueries that \
        you need to answer in order to answer the original question.

        For the given input question, create a query for similarity search and create a query to perform neo4j graph query.
        Here is example:
        Question: Find the articles about the photosynthesis and return their titles.
        Answers:
        sub_query1 : Find articles related to photosynthesis.
        sub_query2 : Return titles of the articles
        """

        return system
    
    def build_chain(self):
        """Build chain function for decompose"""
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{question}"),
            ]
        )

        llm_with_tools = self.llm.bind_tools([SubQuery])
        parser = PydanticToolsParser(tools=[SubQuery])
        query_analyzer = prompt | llm_with_tools | parser

        return query_analyzer
