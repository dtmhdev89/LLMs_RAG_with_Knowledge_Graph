from langchain_ollama import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from pyvis.network import Network
from dotenv import load_dotenv
import os
import asyncio
from src.database_configs.neo4j_graph_db import Neo4jGraphDb

load_dotenv()


class KnowledgeGraphBuilder:
    """Knowledge Graph Builder"""

    OLLAMA_CONNECT_URL = os.getenv("OLLAMA_CONNECT_URL")

    def __init__(self):
        self._llm = ChatOllama(
            model="llama3.1:8b",
            base_url=self.OLLAMA_CONNECT_URL,
            temperature=0,
            streaming=True
        )

        self._graph_transformer = LLMGraphTransformer(llm=self.llm)

        self._neo4j_graph = Neo4jGraphDb().graph

    @property
    def llm(self):
        """llm property"""

        return self._llm

    @property
    def graph_transformer(self):
        """graph_transformer property"""
        return self._graph_transformer

    async def extract_graph_data(self, text):
        """
        Asynchronously extracts graph data from input text using a graph transformer.

        Args:
            text (str): Input text to be processed into graph format.

        Returns:
            list: A list of GraphDocument objects containing nodes and relationships.
        """
        documents = [Document(page_content=text)]
        graph_documents = await self._graph_transformer.aconvert_to_graph_documents(documents)

        return graph_documents

    def visualize_graph(self, graph_documents):
        """
        Visualizes a knowledge graph using PyVis based on the extracted graph documents.

        Args:
            graph_documents (list): A list of GraphDocument objects with nodes and relationships.

        Returns:
            pyvis.network.Network: The visualized network graph object.
        """
        # Create network
        net = Network(
            height="1200px",
            width="100%",
            directed=True,
            notebook=False,
            bgcolor="#222222",
            font_color="white",
            filter_menu=True,
            cdn_resources='remote'
        ) 

        nodes = graph_documents[0].nodes
        relationships = graph_documents[0].relationships

        # Build lookup for valid nodes
        node_dict = {node.id: node for node in nodes}
        
        # Filter out invalid edges and collect valid node IDs
        valid_edges = []
        valid_node_ids = set()
        for rel in relationships:
            if rel.source.id in node_dict and rel.target.id in node_dict:
                valid_edges.append(rel)
                valid_node_ids.update([rel.source.id, rel.target.id])

        # Track which nodes are part of any relationship
        connected_node_ids = set()
        for rel in relationships:
            connected_node_ids.add(rel.source.id)
            connected_node_ids.add(rel.target.id)

        # Add valid nodes to the graph
        for node_id in valid_node_ids:
            node = node_dict[node_id]
            try:
                net.add_node(
                    node.id,
                    label=node.id,
                    title=node.type,
                    group=node.type
                )
            except Exception:
                continue  # Skip node if error occurs

        # Add valid edges to the graph
        for rel in valid_edges:
            try:
                net.add_edge(
                    rel.source.id,
                    rel.target.id,
                    label=rel.type.lower()
                )
            except Exception:
                continue  # Skip edge if error occurs

        # Configure graph layout and physics
        net.set_options("""
            {
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -100,
                        "centralGravity": 0.01,
                        "springLength": 200,
                        "springConstant": 0.08
                    },
                    "minVelocity": 0.75,
                    "solver": "forceAtlas2Based"
                }
            }
        """)

        output_file = "data_outputs/knowledge_graph.html"
        try:
            net.save_graph(output_file)
            print(f"Graph saved to {os.path.abspath(output_file)}")
            return net
        except Exception as e:
            print(f"Error saving graph: {e}")
            return None

    def generate_knowledge_graph(self, text):
        """
        Generates and visualizes a knowledge graph from input text.

        This function runs the graph extraction asynchronously and then visualizes
        the resulting graph using PyVis.

        Args:
            text (str): Input text to convert into a knowledge graph.

        Returns:
            pyvis.network.Network: The visualized network graph object.
        """
        graph_documents = asyncio.run(self.extract_graph_data(text))
        net = self.visualize_graph(graph_documents)

        return graph_documents, net

    def store_graph_to_db(self, graph_documents):
        """Store generated graph documents to Neo4j Graph Database"""

        self._neo4j_graph.add_graph_documents(graph_documents)
