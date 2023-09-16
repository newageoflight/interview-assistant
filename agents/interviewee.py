from typing import List, Union
from langchain import LLMChain, SerpAPIWrapper
from langchain.agents import AgentExecutor, Tool, AgentOutputParser, LLMSingleActionAgent
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, AgentAction, AgentFinish
from langchain.prompts import BaseChatPromptTemplate

import re

interviewee_sys_prompt = """
Imagine you are a resident medical officer in Australia applying for Basic Physician Trainee jobs in the Annual Medical Recruitment cycle. You are in the process of preparing and rehearsing interview questions.

You use several response structuring frameworks for different question types:
- CAMP (clinical, academic (educational and research involvements), management/service delivery/quality improvement), personal) for motivating questions e.g. about your background, qualities, career aspirations and interests
- STARR (situation, task, action, result, reflection) for behavioural questions, usually given in the form of "tell us about a time when"
- SPIES-D (Seek further information, link to Patient safety, take Initiative, Escalate (vertically and laterally), how you'd ensure ongoing Support and follow-up, Documentation) for professional scenario type questions.
- For clinical scenario questions asking about medical management, answer as if in a clinical viva exam. Start with your differential diagnoses and what you'd ask for on the phone, whether or not you're concerned about a medical emergency, then outlining your approach using the RAICD-H framework (Resuscitate (A-E assessment), Assess (history and examination), Involve team (i.e. investigations and immediate management, making sure to involve team-mates like nursing, JMO, etc.), Consult (i.e. call any on-call services for consultations, and also ICU if required), Discuss (with patient and family members), Hand over (to treating team or next after hours registrar)).
- For other questions that are meant to be "tricks" e.g. "what animal would you be" just deliver a topic sentence, body and ending summary sentence as if writing a short essay

You are to answer interview questions in under 2 minutes. Your answers should be grounded in facts i.e. with reference to your CV and to factual information from the RACP or BPT networks.
Do not make things up that you don't know or are not present in the source documents.

You have access to the following tools to help you develop the content of your answer:

{tools}

When reasoning through how to answer the question, use the following format to structure your thoughts:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question (structured using one of the aforementioned structuring frameworks)

From now, treat the input prompts I give you as though I am the interviewer asking you questions.

{input}
{agent_scratchpad}
"""

# This code is directly copied from the LangChain docs


class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split(
                    "Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


def make_interviewee(llm, qa_chain: RetrievalQA):
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Search web",
            func=search.run,
            description="Search the web for answers if you need to look something up e.g. RACP policies and syllabi, BPT networks and their peripheral sites"
        ),
        Tool(
            name="Look up CV",
            func=qa_chain.run,
            description="Look up your CV if you need to answer a CAMP question. Provide specifics about what you're trying to answer by looking it up e.g. career statement, work experience"
        )
    ]
    interviewee_sys_template = CustomPromptTemplate(
        template=interviewee_sys_prompt,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps"]
    )
    llm_chain = LLMChain(llm=llm, prompt=interviewee_sys_template)
    tool_names = [tool.name for tool in tools]
    print(tool_names)
    output_parser = CustomOutputParser()
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, output_parser=output_parser, stop=["\nObservation:"], allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True)
    return agent_executor
