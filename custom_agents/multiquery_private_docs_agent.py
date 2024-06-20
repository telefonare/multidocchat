from typing_extensions import TypedDict
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph
from custom_agents.prompt_formatter import PromptFormatter
import os

from custom_agents.private_docs_agent import private_docs_agent

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
    analysis_choice: str
    query: str
    generation_log: str
    query_plan: str

class multiquery_private_docs_agent:
    def __init__(self,llm):
        self.llm = llm
        self.generate_answer_chain = self._initialize_generate_answer_chain()
        self.generate_query_plan_chain = self._initialize_generate_query_plan_chain()
        self.check_finance_question_chain = self._initialize_check_finance_question_chain()

        self.workflow = StateGraph(GraphState)
        self.workflow.add_node("reject_question", self.reject_question)
        self.workflow.add_node("generate_answer", self.generate_answer)
        self.workflow.add_node("init_agent", self.init_agent)
        self.workflow.add_node("generate_query_plan",self.generate_query_plan)
        self.workflow.add_node("execute_query_plan",self.execute_query_plan)
        
        self.workflow.set_conditional_entry_point(
            self.check_finance_question,
            {
                "analize_doc": "init_agent",
                "reject_question": "reject_question",
            },
        )
        
        self.workflow.add_edge("generate_answer", END)
        self.workflow.add_edge("reject_question", END)
        self.workflow.add_edge("init_agent","generate_query_plan")
        self.workflow.add_edge("generate_query_plan","execute_query_plan")
        self.workflow.add_conditional_edges("execute_query_plan",self.generate_answer)
        
        
        # In[98]:
        
        
        self.local_agent = self.workflow.compile()
        
    def _initialize_generate_answer_chain(self):
        generate_answer_formatter = PromptFormatter("Llama3")
        generate_answer_formatter.init_message("")
        generate_answer_formatter.add_message("""You are an AI assistant for Investment Question Tasks. 
            You have been given the result of a research done by several expert with questions and data added to the original user question for you to generate a clever answer.
            
            Answer the user question with the provided investment data in context. Data is from a financial report from a company.
            If the data provided in context is not relevant to the answer, don't mention the data in context.
            Strictly use the following provided investment data in context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            keep the answer concise, but provide all of the details you can in the form of an informative paragraph.
            Only make direct references to material if provided in the context.
            Only answer questions about investments or finance.
            If there is no data to answer the question just limit to inform the user that there is not enough data to answer the question.
            Provide the answer in the form of a concise report, don´t answer as a person.
            If the data provided does not relate to the answer don´t mention the data, just limit to inform that there is no data to answer. 
        """, "system")
        generate_answer_formatter.add_message("""User Question: {question} 
            Financial data Context: {context}
            Answer: 
        """, "user")
        generate_answer_formatter.close_message("assistant")



        generate_answer_prompt = PromptTemplate(
            template=generate_answer_formatter.prompt,
            input_variables=["question", "context"],
        )
        return generate_answer_prompt | self.llm | StrOutputParser()

    def _initialize_generate_query_plan_chain(self):
        generate_query_plan_formatter = PromptFormatter("Llama3")
        generate_query_plan_formatter.init_message("")
        generate_query_plan_formatter.add_message("""You are an expert in finance and investments.  
            You have received an investment question about a company and you must decompose the user question into several individual questions so you can ask your experts to answer one at a time.
            
            Return a list in JSON with one entry per action needed with a key 'choice' with the name of the choice (i.e. 'ask_private_doc_expert') with no premable or explanation, a key 'query' with you natural language query needed for your collaborator.
            
            If the question of the user needs multiple questions to be asked,analyze the question and  generate one entry per topic, ask one topic at a time..
            
            Your choices are: (by now there is only one expert available, you can call it several times if needed,
            you are only allowed to call the listed experts)

            'ask_private_docs_expert' to make a semantic search in our private documents database, by now it has only the microsoft annual report 2023, it could be useful. He will give you a report

            Question to analyze: {question} 

            Answer only with JSON, no preamble, no headers, no footers, no explanations, just JSON!!
        """, "system")
        generate_query_plan_formatter.close_message("assistant")

        generate_query_plan_prompt = PromptTemplate(
            template=generate_query_plan_formatter.prompt,
            input_variables=["question"],
        )

        return generate_query_plan_prompt | self.llm | JsonOutputParser()

    def _initialize_check_finance_question_chain(self):
        check_finance_question_formatter = PromptFormatter("Llama3")
        check_finance_question_formatter.init_message("")
        check_finance_question_formatter.add_message("""You are an expert at routing a user question to either the analysis stage or to reject the question.
            The user question must be about investments, finance or about information of a company on a given sector or about it's activity and may need to retrieve investment data from
            a semantic database containing a financial report from an important company.
            Give a binary choice 'analize_doc' or 'reject_question' based on the question. 
            Return the JSON with a single key 'choice' with no premable or explanation.
            if the question is not about investmenst or finance you must reject the question.
            
            Question to analyze: {question} 
        """, "system")
        check_finance_question_formatter.close_message("assistant")


        check_finance_question_prompt = PromptTemplate(
            template=check_finance_question_formatter.prompt,
            input_variables=["question"],
        )

        return check_finance_question_prompt | self.llm | JsonOutputParser()

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
        generation = self.generate_answer_chain.invoke({"context": context, "question": question})
        return {"generation": generation}




    def generate_query_plan(self,state):
        info = f"Step: generating plan\n"
        print("#55", info)
        question = state["question"]
        generation_log = state["generation_log"]
        
        
        generation_log += info

        output = self.generate_query_plan_chain.invoke({"question":question})

        return {"query_plan":output}



    def execute_query_plan(self,state):

        private_docs_expert = private_docs_agent(self.llm)

        query_plan = state["query_plan"]
        generation_log = state["generation_log"]
        
        info = f"Step: executing plan\n"
        generation_log += info
        final_output = ""

        for entry in query_plan:
            choice = entry['choice']
            question = entry['query']

            if choice == 'ask_private_docs_expert':
                output = private_docs_expert.ask_question({"question":question})
            else: 
                output = ""

            final_output += output + "\n"
        
        print("#44 ", query_plan)
        
        return {"generation": final_output}
    



    def reject_question(self,state):
        
        print("Step: Rejecting question because is not about investments.")

        generation = "I´m sorry, I cannot process the question because it is not about investments."
        return {"generation": generation}


    def init_agent(self,state):
        print("#init agent")
        return {"context":"","query_plan":""}


# In[96]:


    def can_answer(self,state):
        #print("#2")
        analysis_choice = state["analysis_choice"]
        num_queries = state["num_queries"]
        
        
        if num_queries >= 10:
            return "generate_answer"
        else:
            return analysis_choice    


    def ask_question(self, par_state):
        answer = self.local_agent.invoke(par_state)['generation']
        return answer        




    


    







