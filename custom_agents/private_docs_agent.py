from typing_extensions import TypedDict
#import sqlite3
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

import os
from langchain_groq import ChatGroq
from custom_agents.prompt_formatter import PromptFormatter


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: user question
        generation: LLM generation
        context: results from semantic db so far
    """
    question : str
    generation : str
    context : str
    num_queries: int 
    num_revisions: int
    analysis_choice: str
    query: str
    generation_log: str
    query_historic: str
    next_action: str
    observations: str

# In[6]:
class private_docs_agent:
    def __init__(self,llm):
        self.llm = llm
        self.chroma_client = chromadb.PersistentClient(path="./docsdb2")
        collections = self.chroma_client.list_collections()
        for collection in collections:
            print("#1 ", type(collection), collection.name)
        self.collection = self.chroma_client.get_collection("private_docs")

        self.generate_answer_chain = self._initialize_generate_answer_chain()
        self.analyze_doc_chain = self._initialize_analyze_doc_chain()
        self.reflect_chain = self._initialize_reflect_chain()

        self.workflow = StateGraph(GraphState)
        self.workflow.add_node("generate_answer", self.generate_answer)
        self.workflow.add_node("query_semantic_db", self.query_semantic_db)
        self.workflow.add_node("analyze_doc", self.analyze_doc)
        self.workflow.add_node("init_agent", self.init_agent)
        self.workflow.add_node("reanalize_doc", self.reanalize_doc)
        self.workflow.add_node("add_more_context", self.add_more_context)
        self.workflow.add_node("reflect_on_answer", self.reflect_on_answer)
        #workflow.add_node("can_query",can_query)
        
        self.workflow.set_entry_point("init_agent")
        
        self.workflow.add_edge("generate_answer", END)
        self.workflow.add_edge("init_agent","analyze_doc")
        self.workflow.add_edge("query_semantic_db","reflect_on_answer")
        #workflow.add_edge("analyze_doc", "can_query")
        self.workflow.add_edge("add_more_context","analyze_doc")
        self.workflow.add_edge("reanalize_doc","analyze_doc")
        self.workflow.add_conditional_edges("reflect_on_answer",self.check_retrieval)
        self.workflow.add_conditional_edges("analyze_doc",self.can_query)
        
        
        self.local_agent = self.workflow.compile()
            
    def _initialize_generate_answer_chain(self):
        generate_answer_formatter = PromptFormatter("Llama3")
        generate_answer_formatter.init_message("")
        generate_answer_formatter.add_message("""You are an AI assistant for finance and Investment Questions Tasks. 
            Try to answer the user question with the provided data in context using also the references given. 
            indicate references with a number between parenthesis like this "(1)", then at the end of the report put the meaning of the reference like this "(1): file xxxx". Data is from internal notes.
            If the data provided in context is not relevant to the answer, don't mention the data in context.
            Strictly use the following provided data in context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            keep the answer concise, but provide all of the details you can in the form of an informative paragraph.
            Only make direct references to material if provided in the context and provide the reference of the material to the user.
            Only answer questions about investments or finance field.
            If there is no data to answer the question just limit to inform the user that there is not enough data to answer the question.
            Provide the answer in the form of a concise report, don´t answer as a person.
            If the data provided does not relate to the answer don´t mention the data, just limit to inform that there is no data to answer.
        """, "system")
        generate_answer_formatter.add_message("""Question: {question} 
            Data Context: {context} 
            Answer: 
        """, "user")
        generate_answer_formatter.close_message("assistant")

        generate_answer_prompt = PromptTemplate(
            template=generate_answer_formatter.prompt,
            input_variables=["question", "context"],
        )

        return generate_answer_prompt | self.llm | StrOutputParser()

    def _initialize_analyze_doc_chain(self):
        analyze_doc_formatter = PromptFormatter("Llama3")
        analyze_doc_formatter.init_message("")
        analyze_doc_formatter.add_message("""You are an AI assistant for finance and Investment Questions Tasks. 
            You have received a question about finance or investments field, information in context and query historic.
            
            You must determine if the context data is enough to answer the question,  taking also into account the 'expert observations' you have the following tools, the choices are:
            

            'query_semantic_db' to ask a semantic db the information that you think is relevant to complete the context
            (The vectorstore contains all the internal notes and documents of the company to answer that question)
            Return the JSON with a key 'choice' with no premable or explanation, and a key 'query' with you natural language query needed to search the db if necessary, otherwise leave it blank.
            If the question of the user needs multiple questions to be asked then ask only one, ask one topic at a time, I will provide you the answer and then you will be able to asses all the data and ask again.
            Take into account the query historic, which are the semantic questions already made to the database, do not repeat the questions present in the historic.
            
        
            Question to analyze: {question}.  {observations}
            Query historic (do not repeat the following queries since you have already asked, try another one): {query_historic}
            Data Context: {context} 

        """, "system")

        analyze_doc_formatter.close_message("assistant")
        analyze_doc_prompt = PromptTemplate(
            template=analyze_doc_formatter.prompt,
            input_variables=["question","context","query_historic","observations"],
        )
        
        return analyze_doc_prompt | self.llm | JsonOutputParser()

    def _initialize_reflect_chain(self):
        reflect_formatter = PromptFormatter("Llama3")
        reflect_formatter.init_message("")
        reflect_formatter.add_message("""    You are an expert in finance and investments.
            You have received a question about the finance or investment fields and information in context.
            
            You must determine if the context data is enough to answer the question, you have the following tools, the choices are:
            
            'reanalize_doc' (choose this option if you think the context information is not relevant to answer the user question and it need to be analized again)
            'generate_answer' (if consider there is enough information to elaborate the answer, another agent will complete the task)
            'add_more_context' (if you consider the information in context is correct but it is incomplete and it needs more context )
            Return the JSON with a key 'choice' with no premable or explanation, and a key 'justification' with your explanation about what is needed to retrieve to complete the task, the next agent will complete the task.
            
            Question to analyze: {question} 
            Data Context: {context} 
        ""","system")
        reflect_formatter.close_message("assistant")
        reflect_prompt = PromptTemplate(
            template=reflect_formatter.prompt,
            input_variables=["question","context"],
        )

        return reflect_prompt | self.llm | JsonOutputParser()

    def check_finance_question(self,state):
        """
        route question in order to process it if the question is relative to finance or investments

        Args:
            state (dict): The current graph state

        Returns:
            str: Next node to call
        """

        print("Step: Checking if question is about finance or investments")

        question = state['question']


        output = self.check_finance_question_chain.invoke({"question": question})

        print("Step: Routing to ", output['choice'])

        return output['choice'] 



    def analyze_doc(self, state):
        """
        analyze doc

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("Step: Analizing doc")
        question = state["question"]
        context = state["context"]
        observations = state["observations"]

        query_historic = state["query_historic"]
        #print("invoke anlize doc choice", question, context, query_historic,observations)
        analyze_doc_choice = self.analyze_doc_chain.invoke({"question": question,"context": context,"query_historic":query_historic,"observations":observations})
        print("Analysis choice: ", analyze_doc_choice)
        ### IF analyze_doc_choice
        return {"analysis_choice": analyze_doc_choice["choice"],"query":analyze_doc_choice["query"] }


    # In[90]:


    def generate_answer(self,state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        
        print("Step: Generating Final Response")
        question = state["question"]
        context = state["context"]
        
        # Answer Generation
        print("#7 ", context)
        generation = self.generate_answer_chain.invoke({"context": context, "question": question})
        return {"generation": generation}




    def query_semantic_db(self,state):
        # Retrieve the number of queries from the state
        num_queries = state["num_queries"]
        query_historic = state["query_historic"]
        print("#5")
        
        top_k = 5
        # Extract the user's question from the graph state
        query = state['query']
        query_historic += "\n"+ query
        #print("#1", query)
        #print("#1", num_queries)
        #print("#1", query_historic)
        results = self.collection.query(
        query_texts=[query], # Chroma will embed this for you
        n_results=5 # how many results to return
            )
        print("#6")

        
        report_final = ""
        
        # 
        for i in range(len(results['metadatas'][0])):
            meta = results['metadatas'][0][i]
            doc = results['documents'][0][i]
            referencia = f"page {meta['page']}"
            texto_meta = meta['file_name']
            texto_doc = doc
            
            # Crear la entrada en el reporte final
            report_final += f"Reference number {i+1}: {texto_meta}, Text {referencia}: {texto_doc}\n"
        
        
        
        #print(report_final)

        # Assuming `results` is the dictionary returned from the ChromaDB query
        # and `results['documents']` is a list of strings
        #documents = results['documents'][0]
        
        # Join all the strings in the list into a single string
        #combined_answer = ' '.join(documents)
        
        

        #print("#888 ", state['context'])
        new_context = "" + str(state['context']) + "\n"
        
        # Concatenate the results to the existing context
        new_context += report_final

        # Increment the search counter
        num_queries += 1
        #print("#999 ", new_context, num_queries, query_historic)
        # Return the updated context and the query count
        return {"context": new_context, "num_queries": num_queries,"query_historic":query_historic}



    def reject_question(self,state):
        
        print("Step: Rejecting question because is not about investments.")

        generation = "I´m sorry, I cannot process the question because it is not about investments."
        return {"generation": generation}


# In[95]:


    def init_agent(self,state):
        #print("#init agent")
        return {"num_queries": 0,"query_historic":"","context":"","next_action":""}



    def reflect_on_answer(self,state):
        # Retrieve the necessary information from the state
        context = state["context"]
        question = state["question"]

        reflect_result = self.reflect_chain.invoke({"question": question,"context": context})
        print("#reflection results: ",reflect_result)
        next_action = reflect_result["choice"]
        observations = reflect_result["justification"]

        return {"next_action": next_action,"observations":observations}

    def reanalize_doc(self,state):
        context = ""
        return {"context": context}

    def add_more_context(self,state):
        return

    def can_query(self,state):
        print("#2 can query")
        analysis_choice = state["analysis_choice"]
        num_queries = state["num_queries"]
        print("#3 can query",analysis_choice)
        if analysis_choice == "query_semantic_db":
            print("#4 can query")
            if num_queries >= 2:
                return "generate_answer"
            else:
                return analysis_choice    
        else:
            return analysis_choice



    def check_retrieval(self,state):
        next_action = state["next_action"]

        print("#next action: ", next_action)


        return next_action
    
    def ask_question(self, par_state):
        answer = self.local_agent.invoke(par_state)['generation']
        return answer        



