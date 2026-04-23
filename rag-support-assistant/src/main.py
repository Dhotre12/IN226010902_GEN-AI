import uuid
from langgraph.types import Command
from src.graph import build_graph

def run():
    graph = build_graph()
    print("🤖 RAG Customer Support Assistant (type 'exit' to quit)\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = {"query": query, "intent": "", "retrieved_chunks": [],
                 "confidence": 0.0, "answer": "", "escalated": False, "human_response": None}

        result = graph.invoke(state, config=config)

        # If interrupted (HITL)
        if "__interrupt__" in result or result.get("answer") == "":
            snapshot = graph.get_state(config)
            if snapshot.next:  # paused
                print("\n--- HUMAN AGENT CONSOLE ---")
                print(f"Query: {query}")
                print(f"Retrieved context: {len(snapshot.values.get('retrieved_chunks', []))} chunks")
                human_reply = input("👤 Human agent, type your response: ")
                graph.invoke(Command(resume=human_reply), config=config)

if __name__ == "__main__":
    run()