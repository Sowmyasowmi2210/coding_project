# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import create_react_agent
# from langchain.agents import tool
# from langchain_openai import AzureChatOpenAI
# from langchain_core.messages import BaseMessage
# from typing import TypedDict, Annotated
# import requests

# # -----------------------
# # Step 1: Weather Tool
# # -----------------------
# @tool
# def get_weather(city: str) -> str:
#     """Returns current weather for a given city."""
#     api_key = "YOUR_OPENWEATHERMAP_API_KEY"  # Replace this
#     url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
#     res = requests.get(url)
#     data = res.json()
#     if "main" in data:
#         temp = data["main"]["temp"]
#         desc = data["weather"][0]["description"]
#         return f"The weather in {city} is {temp}°C with {desc}."
#     else:
#         return f"Could not fetch weather for {city}. Please check the city name."

# # -----------------------
# # Step 2: Azure OpenAI LLM
# # -----------------------
# llm = AzureChatOpenAI(
#     openai_api_key="29a42536131c4e288710723d3201d737",               # Replace
#     azure_endpoint="https://wusazeoai02.openai.azure.com/",  # Replace
#     openai_api_version="2023-09-15-preview",
#     deployment_name="gpt-35-turbo"                     # Replace with your deployment name
# )

# # -----------------------
# # Step 3: Create ReAct Agent Node
# # -----------------------
# tools = [get_weather]
# agent_node = create_react_agent(model=llm, tools=tools)

# # -----------------------
# # Step 4: Define State
# # -----------------------
# class AgentState(TypedDict):
#     messages: Annotated[list[BaseMessage], "chat_history"]

# # -----------------------
# # Step 5: Build LangGraph
# # -----------------------
# graph = StateGraph(AgentState)
# graph.add_node("agent", agent_node)
# graph.set_entry_point("agent")
# graph.add_edge("agent", END)

# app = graph.compile()

# # -----------------------
# # Step 6: Run the Graph
# # -----------------------
# response = app.invoke(
#     {"messages": [{"role": "user", "content": "What's the weather in Chennai?"}]}
# )

# # -----------------------
# # Step 7: Output
# # -----------------------
# print(response["messages"][-1].content)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain.agents import tool
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
import wikipedia


@tool
def search_wikipedia(topic: str) -> str:
    """Searches Wikipedia and returns a summary of the topic."""
    try:
        summary = wikipedia.summary(topic, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Topic '{topic}' is ambiguous. Try one of these: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return f"No page found for '{topic}'."
    except Exception as e:
        return f"An error occurred: {str(e)}"


llm = AzureChatOpenAI(
    openai_api_key="29a42536131c4e288710723d3201d737",
    azure_endpoint="https://wusazeoai02.openai.azure.com/",
    openai_api_version="2023-09-15-preview",
    deployment_name="gpt-35-turbo"
)


tools = [search_wikipedia]
agent_node = create_react_agent(model=llm, tools=tools)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "chat_history"]

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.set_entry_point("agent")
graph.add_edge("agent", END)

app = graph.compile()
response = app.invoke(
    {
        "messages": [
            {"role": "system", "content": "You can use the tool 'search_wikipedia' to answer questions by retrieving information from Wikipedia."},
            {"role": "user", "content": "Tell me about Taj Mahal"}
        ]
    }
)
print(response["messages"][-1].content)

