from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
import os

load_dotenv()


class Neo4jGraphDb:
    """Neo4j Graph Database"""

    def __init__(self):
        self._graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )

    @property
    def graph(self):
        """graph property"""
        return self._graph
