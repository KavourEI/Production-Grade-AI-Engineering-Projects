from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import Annotated
from langgraph.graph.message import add_messages
import sqlite3
from typing import TypedDict

class SummarizedState(MessagesState):
    summary: str

llm = ChatOllama(model='llama3.1:8b')

def call_model(state:SummarizedState)->SummarizedState:
    """Prepare and update, if needed, the running summary as a system message, then call LLM."""
    messages = state["messages"]
    summary = state.get("summary", "")

    if summary:
        system_msg = SystemMessage(content=f"Summary of the conversation so far:\n{summary}")
        messages = [system_msg] + messages

    response = llm.invoke(messages)
    print(response.content)
    return {"messages": [response]}

SUMMARIZE_AFTER = 6

def summarize_conversation(state: SummarizedState) -> dict:
    """Ask LLm to re-summarize the whole conversation, then delete all but the last 2 messages to keep the window short"""
    summary = state.get("summary", "")
    messages = state["messages"]

    if summary:
        prompt = (
            f"This is a summary of the conversation so far:\n{summary}\n\n"
            "Extend the summary to include the new messages above, "
            "keeping it concise."
        )
    else:
        prompt = "Summarize the conversation above in a concise paragraph."
    
    summary_response = llm.invoke(messages + [HumanMessage(content=prompt)])
    new_summary = summary_response.content

    keep_count = 4
    messages_to_delete = [
        RemoveMessage(id=m.id) for m in messages[:-keep_count]
        if m.id is not None          # guard: skip any message without an id
    ]

    return {
        "summary": new_summary,
        "messages": messages_to_delete
    }

def should_summarize(state: SummarizedState) -> str:
    """Routing to summarization node when history grows too long."""
    if len(state["messages"]) > SUMMARIZE_AFTER:
        return "summarize"
    else:
        return "end"
        

builder = StateGraph(SummarizedState)

builder.add_node("call_model", call_model)
builder.add_node("summarize_conversation", summarize_conversation)

builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model",
    should_summarize,
    {
        "summarize": "summarize_conversation",
        "end": END
    }
)
builder.add_edge("summarize_conversation", END)

conn=sqlite3.connect("chats.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": [{"role": "user", "content": "Hi there!"}]}, config=config)