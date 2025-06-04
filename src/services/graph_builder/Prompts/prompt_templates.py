import os
from langchain.vectorstores import Chroma
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.example_selectors import MaxMarginalRelevanceExampleSelector
from src.services.graph_builder.Prompts.prompt_examples import examples

# Import Custom Libraries
from src.services.graph_builder.Graph.state import GraphState


class PromptTemplates:
    """Prompt Template"""

    EMBEDDING_MODEL = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    def __init__(self) -> None:
        # Instantiate a example selector
        self.example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
            examples=examples,
            embeddings=self.EMBEDDING_MODEL,
            vectorstore_cls=Chroma,
            k=5,
        )

        # Configure a formatter
        self.example_prompt = PromptTemplate(
            input_variables=["question", "query"],
            template="Question: {question}\nCypher query: {query}"
        )

    def create_few_shot_prompt(self):
        '''Create a prompt template without context variable. The suffix provides dynamically selected prompt examples using similarity search'''
        
        prefix = """
        Task:Generate Cypher statement to query a graph database.
        Instructions:
        Use only the provided relationship types and properties in the schema.
        Do not use any other relationship types or properties that are not provided.

        Note: Do not include any explanations or apologies in your responses.
        Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
        Do not include any text except the generated Cypher statement.

        Examples: Here are a few examples of generated Cypher statements for particular questions:
        """

        few_shot_prompt = FewShotPromptTemplate(
            example_selector=self.example_selector,
            example_prompt=self.example_prompt,
            prefix=prefix,
            suffix="Question: {question}, \nCypher Query: ",
            input_variables=["question","query"],
        )

        return few_shot_prompt

    def create_few_shot_prompt_with_context(self, state: GraphState):
        '''Create a prompt template with context variable. The context variable will be based on the output from vector qa chain'''
        '''The output of vector qa is list of node ids against which to perform graph query'''
        
        context = state["article_ids"]
        
        prefix = f"""
        Task:Generate Cypher statement to query a graph database.
        Instructions:
        Use only the provided relationship types and properties in the schema.
        Do not use any other relationship types or properties that are not provided.

        Note: Do not include any explanations or apologies in your responses.
        Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
        Do not include any text except the generated Cypher statement.
        
        A context is provided from a vector search in a form of tuple ('a..', 'W..') 
        Use the second element of the tuple as a node id, e.g 'W..... 
        Here are the contexts: {context}

        Using node id from the context above, create cypher statements and use that to query with the graph.
        Examples: Here are a few examples of generated Cypher statements for some question examples:
        """

        few_shot_prompt = FewShotPromptTemplate(
            example_selector=self.example_selector,
            example_prompt=self.example_prompt,
            prefix=prefix,
            suffix="Question: {question}, \nCypher Query: ",
            input_variables=["question", "query"],
        ) 
        return few_shot_prompt
