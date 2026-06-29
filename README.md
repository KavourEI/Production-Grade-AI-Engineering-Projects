# Production-Grade-AI-Engineering-Projects

## Table of Content

1. Stateful Conversational Chatbot
<!-- 2. Basic RAG
3. Advanced/Hybrid RAG with Reranking
4. Structured LLM Pipeline (Extraction & Summarization)
5. Evaluation & Observability Harness
6. Single Tool - Using Agent (ReAct)
7. Agentic RAG (Self-Correcting Retrival)
8. Multi-Agent System
9. QLoRA Fine-Tuning + Local Serving -->

### TL;DR

* This roadmap converts 9 production AI projects into sequential, searchable developer checklists (15-25 steps each) grouped into **Setup**, **Core Implementation**, **Production Layer**, and **Evaluation** phases, targeting the stack: *Python*, *LangChain/LangGraph*, *Ollama*, *FastAPI*, *Pydantic*, *Chroma/Qdrant*, *Langfuse*, *RAGAS*, *DeepEval* and *Unsloth*.

* Each step is a single concrete action a developer can search and implement (e.g. "Compile the StateGraph with a ```PostgresSaver``` checkpointer", "Run ```ollama create -f Modelfile```, "Fan out map steps with LangGraph's ```Send``` API").

## Project 1: Stateful Conversational Chatbot

### - Setup:

1. Create a Python project with ```uv``` or ```venv``` and install ```langgraph```, ```langchain```, ```langchain-ollama```,```fastapi```,```uvicorn```,```pydantic```.
2. Install and start Ollama locally; pull a chat model with ```ollama pull llama3.1```(or similar).
3. Verify the model responds via ```ollama run llama3.1``` before writting code.
4. Define the LLM in code with ```ChatOllama(model="llama3.1")``` from ```langchain_ollama```.


### - Core Implementation:

1. Import ```StateGraph```, ```MessagesState```, ```START```, ```END``` from ```langgraph.graph```.
2. Define a ```call_model(state: MessagesState)``` node that calls `model.invoke(state["messages"])` and returens `{"messages": response}`.
3. Build the graph `builder = StateGraph(MessagesState)`, `add_node`, `add_edge(START, "call_model")`.
4. Add an `InMemorySaver` checkpointer from `langgraph.checkpoint.memory` and `compile(checkpointer=...)`.
5. Invoke the graph with a `config={"configurable": {"thread_id":"1"}}` to test multi-turn memory.
6. Swap the checkpointer `SqliteSaver` ( `langgraph.checkpoint.sqlite` ) for durable local persistence.
7. Add a conversation-summarization node: subclass `MessagesState` with a `summary: str` field and use `Remove Message` to trim history.

### - Production Layer:

1. Upgrade to `AsyncPostgresSaver` from `langgraph.checkpoint.postgres.aio` and call `await checkpointer.setup()` once.
2.  Create a FastAPI app and a `POST /chat` endpoint accepting a Pydantic request model with `thread_id` and `message`.
3. Implement token streaming with `graph.stream(..., stream_mode="messages)` and wrap it in a FastAPI `StreamingResponse` with `media_type="text/event-stream"`.
4. Format each chunk as a Server-Sent Event ( `yield f"data: {json}\n\n"` ).
5. Add a `GET /history/{thread_id}` endpoint using `graph.get_state(config)`.
6. Run the server with `uvicorn` and confirm streaming in `/docs` or via `curl -N`.
 

### - Evaluation:

1. Add structured logging of `tread_id`, latency, and token counts per request.
2. Write a multi-turn test that asserts the bot recalls a fact stated in an earlier turn (same `thread_id`).
3. Add graceful handling for Ollama connection errors and context-window overflow.

---