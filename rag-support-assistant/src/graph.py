from typing import TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.retriever import retrieve
from src.config import CONFIDENCE_THRESHOLD, GOOGLE_API_KEY

# -------- STATE --------
class AgentState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[dict]
    confidence: float
    answer: str
    escalated: bool
    human_response: Optional[str]

# -------- LLM --------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# -------- NODES --------
def input_node(state: AgentState) -> AgentState:
    q = state["query"].lower()
    if any(k in q for k in ["refund", "cancel", "complaint", "angry", "legal", "speak to human", "agent"]):
        intent = "COMPLAINT"
    elif any(k in q for k in ["how", "what", "where", "when", "why"]):
        intent = "FAQ"
    elif len(q.split()) > 25:
        intent = "COMPLEX"
    else:
        intent = "FAQ"
    print(f"🧭 Intent detected: {intent}")
    return {"intent": intent}

def retrieve_node(state: AgentState) -> AgentState:
    chunks, avg_score = retrieve(state["query"])
    print(f"🔍 Retrieved {len(chunks)} chunks | avg score: {avg_score:.2f}")
    return {"retrieved_chunks": chunks, "confidence": avg_score}

def generate_node(state: AgentState) -> AgentState:
    context = "\n\n".join([c["text"] for c in state["retrieved_chunks"]])
    prompt = ChatPromptTemplate.from_template("""
You are a helpful customer support assistant. Answer ONLY using the context below.
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer (concise, friendly):
""")
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": state["query"]})
    return {"answer": response.content, "escalated": False}

def hitl_node(state: AgentState) -> AgentState:
    print("\n🚨 ESCALATING TO HUMAN AGENT 🚨")
    human_input = interrupt({
        "query": state["query"],
        "intent": state["intent"],
        "confidence": state["confidence"],
        "reason": "Low confidence or sensitive intent"
    })
    return {"answer": human_input, "escalated": True, "human_response": human_input}

def output_node(state: AgentState) -> AgentState:
    print("\n" + "="*60)
    print(f"📤 FINAL ANSWER ({'👤 HUMAN' if state['escalated'] else '🤖 AI'}):")
    print(state["answer"])
    print("="*60)
    return state

# -------- ROUTER --------
def route_decision(state: AgentState) -> Literal["generate", "escalate"]:
    if not state["retrieved_chunks"]:
        return "escalate"
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        return "escalate"
    if state["intent"] in ["COMPLAINT", "COMPLEX"]:
        return "escalate"
    return "generate"

# -------- BUILD GRAPH --------
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("input", input_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("hitl", hitl_node)
    g.add_node("output", output_node)

    g.add_edge(START, "input")
    g.add_edge("input", "retrieve")
    g.add_conditional_edges("retrieve", route_decision, {
        "generate": "generate",
        "escalate": "hitl"
    })
    g.add_edge("generate", "output")
    g.add_edge("hitl", "output")
    g.add_edge("output", END)

    return g.compile(checkpointer=MemorySaver())