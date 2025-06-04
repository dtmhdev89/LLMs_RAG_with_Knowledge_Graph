import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jVector
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings

load_dotenv()


class Index:
    """Index Support Methods"""

    EMBEDDING_MODEL = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    def __init__(self):
        pass

    def get_neo4j_vector_index(self):
        ''' Create vector for article title and abstract and Instantiate Neo4j vector from graph'''
        neo4j_vector_index = Neo4jVector.from_existing_graph(
            embedding=self.EMBEDDING_MODEL,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name='title_abstract_vector',
            node_label='Article',
            text_node_properties=['title', 'abstract'],
            embedding_node_property='embedding_vectors',
        )
        
        return neo4j_vector_index

    def get_neo4j_title_vector_index(self): 
        '''Create a title vector and Instantiate Neo4j vector from graph'''
        
        neo4j_title_vector_index = Neo4jVector.from_existing_graph(
            embedding=self.EMBEDDING_MODEL,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name='title_vector',
            node_label='Title',
            text_node_properties=['text'],
            embedding_node_property='embedding_vectors',
        )

        return neo4j_title_vector_index

    def get_neo4j_abstract_vector_index(self): 
        ''' Create an abstract vector and Instantiate Neo4j vector from graph'''

        neo4j_abstract_vector_index = Neo4jVector.from_existing_graph(
            embedding=self.EMBEDDING_MODEL,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name='abstract_vector',
            node_label='Abstract',
            text_node_properties=['text'],
            embedding_node_property='embedding_vectors',
        )

        return neo4j_abstract_vector_index

    def get_neo4j_topic_vector_index(self): 
        '''Create a topic vector and Instantiate Neo4j vector from graph'''

        neo4j_topic_vector_index = Neo4jVector.from_existing_graph(
            embedding=self.EMBEDDING_MODEL,
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            index_name='topic_vector',
            node_label='Topic',
            text_node_properties=['text'],
            embedding_node_property='embedding_vectors',
        )

        return neo4j_topic_vector_index
