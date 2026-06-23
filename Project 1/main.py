from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

llm = ChatOllama(
    model = 'llama3.1:8b',
)

def call_model(state: MessagesState) -> MessagesState:
    response = llm.invoke(state["messages"])
    print(response.content)
    return state

builder = StateGraph(MessagesState)

builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge('call_model', END)


import sqlite3
conn =sqlite3.connect('chats.db', check_same_thread=False)
# checkpointer = InMemorySaver()
new_checkpointer = SqliteSaver(conn=conn)

graph = builder.compile(checkpointer=new_checkpointer)
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": [{"role": "user", "content": "Hi there!"}]}, config=config)
graph.invoke({"messages": [{"role": "user", "content": "My name is Themis"}]}, config=config)
# Subsequent runs with the same thread_id will have access to the memory
graph.invoke({"messages": [{"role": "user", "content": "What was my previous message?"}]}, config=config)
graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config=config)