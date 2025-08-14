from langgraph.graph import StateGraph
from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from typing import TypedDict
import os

# --------------------------
# 1. Define Shared State
# --------------------------
class GraphState(TypedDict):
    query: str
    relevant_docs: str
    reasoning: str

# --------------------------
# 2. Load & Prepare Data
# --------------------------
# Load documents
loader = TextLoader("sample.txt")  # replace with your file
docs = loader.load()

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embedding model
embedding_model = OpenAIEmbeddings()  # or AzureOpenAIEmbeddings if using Azure

# Create FAISS index
vectorstore = FAISS.from_documents(chunks, embedding_model)

# --------------------------
# 3. Define Nodes
# --------------------------

# Node: Vector search
def faiss_search_node(state: GraphState) -> dict:
    query = state["query"]
    results = vectorstore.similarity_search(query, k=3)
    combined = "\n".join([doc.page_content for doc in results])
    return {"relevant_docs": combined}

# Node: Reasoning using LLM
llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo",
    api_key="29a42536131c4e288710723d3201d737",
    api_version="2023-09-15-preview",
    azure_endpoint="https://wusazeoai02.openai.azure.com/"
)

def reasoning_node(state: GraphState) -> dict:
    docs = state["relevant_docs"]
    query = state["query"]
    prompt = f"Given the query: {query}, and the following context:\n{docs}\nAnswer the query clearly."
    result = llm.invoke(prompt)
    return {"reasoning": result.content}

# Node: Final output
def final_node(state: GraphState) -> dict:
    print("Final Output:")
    print(" Query:", state["query"])
    print(" Retrieved Docs:", state["relevant_docs"])
    print(" Reasoning:", state["reasoning"])
    return {}

# --------------------------
# 4. Build LangGraph
# --------------------------
builder = StateGraph(GraphState)

builder.add_node("faiss_search", faiss_search_node)
builder.add_node("reasoning", reasoning_node)
builder.add_node("final", final_node)

builder.set_entry_point("faiss_search")
builder.add_edge("faiss_search", "reasoning")
builder.add_edge("reasoning", "final")
builder.set_finish_point("final")

graph = builder.compile()

# --------------------------
# 5. Run Graph
# --------------------------
graph.invoke({"query": "What is LangChain used for?"})