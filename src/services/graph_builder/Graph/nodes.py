import os
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.services.graph_builder.Graph.state import GraphState
from src.services.graph_builder.Chains.decompose import Decompose
from src.services.graph_builder.Chains.vector_graph_chain import VectorGraphChain
from src.services.graph_builder.Chains.graph_qa_chain import GraphQaChain

from src.services.graph_builder.Tools.parse_vector_search import DocumentModel
from src.services.graph_builder.Prompts.prompt_templates import PromptTemplates


class Nodes:
    """Nodes for the graph"""

    def __init__(self):
        self._query_analyzer = Decompose(
            provider='ollama',
            model_name='llama3.1:8b'
        ).build_chain()
        self._vector_graph_chain = VectorGraphChain(
            provider='ollama'
        ).get_vector_graph_chain()
        self._prompt_templates = PromptTemplates()
        self._graph_qa_chain = GraphQaChain(provider='ollama')

    def decomposer(self, state: GraphState):
        '''
        Returns a dictionary of at least one of the GraphState
        Decompose a given question to sub-queries'
        '''

        question = state["question"]
        subqueries = self._query_analyzer.invoke(question)

        return {"subqueries": subqueries, "question": question}

    def vector_search(self, state: GraphState):
        """Vector search node
        Returns a dictionary of at least one of the GraphState
        Perform a vector similarity search and return article id as a parsed outpu
        """

        question = state["question"]
        queries = state["subqueries"]
    
        chain_result = self._vector_graph_chain.invoke({
                "input": queries[0].sub_query
            },
        )

        # Convert the result to a list of DocumentModel instances
        documents = [
            DocumentModel(**doc.dict())
            for doc in chain_result['context']
        ]
        extracted_data = [
            {
                "title": doc.extract_title(),
                "article_id": doc.metadata.article_id
            }
            for doc in documents
        ]
        article_ids = [
            ("article_id", doc.metadata.article_id)
            for doc in documents
        ]
        
        return {
            "article_ids": article_ids,
            "documents": extracted_data,
            "question": question,
            "subqueries": queries
        }

    def prompt_template(self, state: GraphState):
        '''
        Returns a dictionary of at least one of the GraphState
        Create a simple prompt tempalate for graph qa chain
        '''
        
        question = state["question"]

        prompt = self._prompt_templates.create_few_shot_prompt()
        
        return {"prompt": prompt, "question": question}

    def graph_qa(self, state: GraphState):
        '''
        Returns a dictionary of at least one of the GraphState
        Invoke a Graph QA Chain
        '''
        
        question = state["question"]
        
        graph_qa_chain = self._graph_qa_chain.get_graph_qa_chain(state)
        
        result = graph_qa_chain.invoke(
            {
                #"context": graph.schema, 
                "query": question,
            },
        )
        return {"documents": result, "question": question}
    
    def prompt_template_with_context(self, state: GraphState):
        '''
        Returns a dictionary of at least one of the GraphState
        Create a dynamic prompt template for graph qa with context chain
        '''
        
        question = state["question"]
        queries = state["subqueries"]

        # Create a prompt template
        prompt_with_context = self._prompt_templates.create_few_shot_prompt_with_context(state)
        
        return {
            "prompt_with_context": prompt_with_context,
            "question": question,
            "subqueries": queries
        }

    def graph_qa_with_context(self, state: GraphState):
        '''
        Returns a dictionary of at least one of the GraphState
        Invoke a Graph QA chain with dynamic prompt template
        '''
        
        queries = state["subqueries"]
        prompt_with_context = state["prompt_with_context"]

        # Instantiate graph_qa_chain_with_context
        # Pass the GraphState as 'state'. This chain uses state['prompt'] as input argument
        graph_qa_chain = self._graph_qa_chain.get_graph_qa_chain_with_context(state)
        
        result = graph_qa_chain.invoke(
            {
                "query": queries[1].sub_query,
            },
        )
        return {
            "documents": result,
            "prompt_with_context": prompt_with_context,
            "subqueries": queries
        }
