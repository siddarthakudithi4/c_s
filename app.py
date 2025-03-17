import streamlit as st
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API Keys
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
groq_llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

# ================== Define Workflow State ==================
class AgentState(TypedDict):
    input: str
    tool_outputs: Annotated[Sequence[str], operator.add]
    next_node: str
    final_output: str

# ================== Define Nodes ==================
def route_node(state: AgentState) -> dict:
    return {"next_node": "general_llm"}

def general_llm_node(state: AgentState) -> dict:
    try:
        response = groq_llm.invoke([{"role": "user", "content": state["input"]}])
        return {"tool_outputs": [response.content]}
    except Exception as e:
        return {"tool_outputs": [f"Error: {str(e)}"]}

def post_processing_node(state: AgentState) -> dict:
    valid_responses = [msg for msg in state["tool_outputs"] if msg.strip()]
    return {"final_output": valid_responses[0] if valid_responses else "No response"}

# ================== Create Workflow Graph ==================
workflow = StateGraph(AgentState)

workflow.add_node("route", route_node)
workflow.add_node("general_llm", general_llm_node)
workflow.add_node("post_processing", post_processing_node)

workflow.add_conditional_edges(
    "route",
    lambda state: state["next_node"],
    {"general_llm": "general_llm"}
)

workflow.add_edge("general_llm", "post_processing")
workflow.add_edge("post_processing", END)

workflow.set_entry_point("route")
app = workflow.compile()

# ================== Streamlit App Setup ==================
st.set_page_config(page_title="Amazon Assistant", layout="wide")
st.title("Amazon Customer Support Chatbot")

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process query through the chatbot
    response = app.invoke({
        "input": prompt,
        "tool_outputs": [],
        "next_node": "",
        "final_output": ""
    }).get("final_output", "No response")
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(response)