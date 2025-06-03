from langchain_neo4j import Neo4jGraph


class Neo4jGraphDb:
    """Neo4j Graph Database"""

    def __init__(self):
        self._graph = Neo4jGraph()

    @property
    def graph(self):
        """graph property"""
        return self._graph
