from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="GLUG NIT Durgapur",
    page_icon="🤖",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    .main-header p {
        color: #8b8fa8;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .stChatMessage { border-radius: 12px; margin-bottom: 0.5rem; }
    .stChatInputContainer { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🐧 GLUG Chatbot</h1>
    <p>GNU/Linux Users Group · NIT Durgapur</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Build LangGraph (cached so it's built only once) ─────────
@st.cache_resource
def build_chatbot():
    class ChatState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    def chat_node(state: ChatState):
        return {'messages': [llm.invoke(state['messages'])]}

    g = StateGraph(ChatState)
    g.add_node('chat_node', chat_node)
    g.add_edge(START, 'chat_node')
    g.add_edge('chat_node', END)
    return g.compile()

chatbot = build_chatbot()

# ── Session state (conversation history) ─────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []         

if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = []      

# ── Render chat history ───────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🐧" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about GLUG, NIT Durgapur..."):

    # Show user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Save to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.lc_messages.append(HumanMessage(content=prompt))

    # Get response with spinner
    with st.chat_message("assistant", avatar="🐧"):
        with st.spinner("Thinking..."):
            result = chatbot.invoke({"messages": st.session_state.lc_messages})
            reply = result["messages"][-1].content

        st.markdown(reply)

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.lc_messages.append(AIMessage(content=reply))

# ── Empty state hint ──────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; color:#8b8fa8; padding: 3rem 1rem;">
        <p style="font-size:2.5rem">💬</p>
        <p>Ask me about GLUG events, Linux, open source, or anything about NIT Durgapur!</p>
    </div>
    """, unsafe_allow_html=True)