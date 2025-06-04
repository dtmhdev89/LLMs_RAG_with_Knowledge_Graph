from dotenv import load_dotenv
load_dotenv()
from typing import Literal
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_neo4j import Neo4jGraph

from src.services.graph_builder.Graph.state import GraphState


class GraphQaChain:
    def __init__(self, provider: Literal['ollama', 'openai']):
        if provider == 'ollama':
            self.llm = ChatOllama(
                model="llama3.1:8b",
                temperature=0,
                base_url=os.getenv("OLLAMA_CONNECT_URL")
            )
        elif provider == "openai":
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo", 
                temperature=0,
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        
        self.graph = Neo4jGraph(
            url=os.environ.get('NEO4J_URI'),
            username=os.environ.get('NEO4J_USERNAME'),
            password=os.environ.get('NEO4J_PASSWORD')
        )

    def get_graph_qa_chain(self, state: GraphState):
        """Create a Neo4j Graph Cypher QA Chain"""
        
        prompt = state["prompt"]
        
        graph_qa_chain = GraphCypherQAChain.from_llm(
                cypher_llm=self.llm, #should use gpt-4 for production
                qa_llm=self.llm,
                validate_cypher=True,
                graph=self.graph,
                verbose=True,
                cypher_prompt=prompt,
                # return_intermediate_steps = True,
                return_direct=True,
                allow_dangerous_requests=True,
            )

        return graph_qa_chain

    def get_graph_qa_chain_with_context(self, state: GraphState):
        """
        Create a Neo4j Graph Cypher QA Chain. Using this as GraphState so it can access state['prompt']
        """
        
        prompt_with_context = state["prompt_with_context"]
        
        graph_qa_chain = GraphCypherQAChain.from_llm(
            cypher_llm=self.llm, #should use gpt-4 for production
            qa_llm=self.llm,
            validate_cypher=True,
            graph=self.graph,
            verbose=False,
            cypher_prompt=prompt_with_context,
            # return_intermediate_steps = True,
            return_direct=True,
            allow_dangerous_requests=True,
        )

        return graph_qa_chain
