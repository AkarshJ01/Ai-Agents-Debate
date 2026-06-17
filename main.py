import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage , ToolMessage


from langchain_tavily import TavilySearch


LLM = ['gpt-oss:20b','gpt-oss:20b','gpt-oss:20b']

llm1 = ChatOllama(temperature=0, model= LLM[0])
llm2 = ChatOllama(temperature=0, model= LLM[1])
judge = ChatOllama(temperature=0, model= LLM[2])


tavily_tool = TavilySearch()

for_subject = llm1.bind_tools([tavily_tool])
against_subject = llm2.bind_tools([tavily_tool])


def run_agents(prompt:str):

    message_for = [
        SystemMessage(content="You are an objective research agent. You MUST use your web search tool to find live, real-world data, historical statistics, and evidence that supports the user's statement. Rely ONLY on the tool results, not your training data."),
        HumanMessage(content=f"Find supporting arguments for: {prompt}")
    ]

    message_against = [
        SystemMessage(content="You are an objective research agent. You MUST use your web search tool to find live, real-world data, counter-statistics, and expert criticisms that disprove the user's statement. Rely ONLY on the tool results, not your training data."),
        HumanMessage(content=f"Find opposing arguments against: {prompt}")
    ]


    for_response = for_subject.invoke(message_for)
    against_response = against_subject.invoke(message_against)


    message_for.append(for_response)
    message_against.append(against_response)

    final_response_for = None
    final_response_against = None

    if for_response.tool_calls and against_response.tool_calls:
        for tool_call in for_response.tool_calls:
            if "tavily" in tool_call["name"].lower():
                # Doing the Tavily Tool call to get output
                tool_output_1 = tavily_tool.invoke(tool_call["args"])
                                
                # CHANGED: Feed as HumanMessage so local Ollama template reads it cleanly
                message_for.append(
                    HumanMessage(
                        content=f"Here are the web search results to construct your answer:\n{str(tool_output_1)}"
                    )
                )
        
        # CHANGED: Call the raw model (llm1) instead of for_subject so it outputs text instead of another tool call
        final_response_for = llm1.invoke(message_for)
        print("\nStatement For the Subject :\n", final_response_for.content)


        for tool_call in against_response.tool_calls:
            if "tavily" in tool_call["name"].lower():
                # Doing the Tavily Tool call to get output
                tool_output_2 = tavily_tool.invoke(tool_call["args"])
                                
                # CHANGED: Feed as HumanMessage so local Ollama template reads it cleanly
                message_against.append(
                    HumanMessage(
                        content=f"Here are the web search results to construct your answer:\n{str(tool_output_2)}"
                    )
                )
        
        # CHANGED: Call the raw model (llm2) instead of against_subject so it outputs text instead of another tool call
        final_response_against = llm2.invoke(message_against)
        print("\nStatement Against the Subject :\n", final_response_against.content)



    else:
        print("One or both of the ai models didn't make any tool calls")
        return        


    message_judge = [
            SystemMessage(content="You are an elite, completely neutral AI Referee. Review the provided arguments. Evaluate both sides strictly based on the quality and relevance of the facts they fetched. You must provide a clear winning/losing percentage breakdown totaling 100% (e.g., 65% For / 35% Against) and a brief justification."),
            HumanMessage(content=f"""
    The debate topic was: "{prompt}"

    Arguments presented FOR:
    {final_response_for.content}

    Arguments presented AGAINST:
    {final_response_against.content}
    """)
        ]

    final_verdict = judge.invoke(message_judge)
    print("\nFinal Verdict From the Judge :\n", final_verdict.content)



if __name__ == "__main__":
    run_agents("Child Labour is bad for society")