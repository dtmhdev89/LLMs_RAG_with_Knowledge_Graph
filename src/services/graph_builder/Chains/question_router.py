import os
from dotenv import load_dotenv
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal['vector search', 'graph query'] = Field(
        ...,
        description="Given a user question choose to route it to vectorstore or graphdb."
    )


class QuestionRouter:
    OLLAMA_CONNECT_URL = os.getenv("OLLAMA_CONNECT_URL")
    
    def __init__(
        self,
        provider: Literal['ollama', 'openai'],
        model_name: str
    ) -> None:
        if provider == 'ollama':
            self.llm = ChatOllama(
                model=model_name,
                temperature=0,
                base_url=self.OLLAMA_CONNECT_URL
            )
        elif provider == 'openai':
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0
            )
    
    @property
    def system_prompt(self):
        """system prompt"""
        
        system = """You are an expert at routing a user question to perform vector search or graph query. 
        The vector store contains documents related article title, abstracts and topics. Here are three routing situations:
        If the user question is about similarity search, perform vector search. The user query may include term like similar, related, relvant, identitical, closest etc to suggest vector search. For all else, use graph query.

        Example questions of Vector Search Case: 
            Find articles about photosynthesis
            Find similar articles that is about oxidative stress
            
        Example questions of Graph DB Query: 
            MATCH (n:Article) RETURN COUNT(n)
            MATCH (n:Article) RETURN n.title

        Example questions of Graph QA Chain: 
            Find articles published in a specific year and return it's title, authors
            Find authors from the institutions who are located in a specific country, e.g Japan
        """

        return system
    
    def build_router(self):
        """Question router chain"""

        structured_llm_router = self.llm.with_structured_output(RouteQuery)

        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{question}")
            ]
        )
        question_router = route_prompt | structured_llm_router

        return question_router

