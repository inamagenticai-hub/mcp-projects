import streamlit as st
import yagmail
import os
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="MCP Agent", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #0d0f18; }
    [data-testid="stSidebar"] { background: #0f1117; border-right: 1px solid #1e2130; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .user-msg { background:#1a1d2e; border-left:3px solid #6366f1; border-radius:8px; padding:14px 18px; margin:10px 0; color:#e2e8f0; }
    .assistant-msg { background:#111827; border-left:3px solid #10b981; border-radius:8px; padding:14px 18px; margin:10px 0; color:#d1fae5; }
    .stButton > button { background:#6366f1; color:white; border:none; border-radius:8px; font-weight:600; }
    .stButton > button:hover { background:#4f46e5; }
    h1,h2,h3 { color:#f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Tools ──────────────────────────────────────────────────────────────────
@tool
def add(a: float, b: float) -> float:
    """
    ALWAYS use this tool when user asks to add, sum, or calculate numbers.
    Args:
        a: first number
        b: second number
    Returns the sum of a and b.
    """
    return int(a) + int(b)


@tool
def greet(name: str) -> str:
    """
    ALWAYS use this tool when user wants to greet someone or says 'greet my friend X'.
    Args:
        name: full name of the person to greet
    Returns a greeting message for that person.
    """
    return f"Hello, {name}! Hope you are doing well! 😊"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    ALWAYS use this tool when user wants to send an email.
    Args:
        to: recipient email address
        subject: email subject
        body: email body text
    Returns confirmation of email sent.
    """
    sender = os.getenv("SENDER_EMAIL")
    password = os.getenv("SENDER_PASSWORD")
    if not sender or not password:
        return "❌ Email credentials missing in Streamlit secrets."
    try:
        yag = yagmail.SMTP(user=sender, password=password)
        yag.send(to=to, subject=subject, contents=body)
        return f"✅ Email sent to {to} | Subject: '{subject}'"
    except Exception as e:
        return f"❌ Failed: {str(e)}"


tools = [send_email, add, greet]


# ─── Agent ──────────────────────────────────────────────────────────────────
@st.cache_resource
def build_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=(
            "You are a helpful assistant. You have access to tools.\n"
            "IMPORTANT RULES:\n"
            "1. ALWAYS use the 'greet' tool when user wants to greet someone.\n"
            "2. ALWAYS use the 'add' tool when user wants to add numbers.\n"
            "3. ALWAYS use the 'send_email' tool when user wants to send email.\n"
            "4. NEVER answer from memory when a tool is available.\n"
            "5. Show the exact output returned by the tool in your response."
        )
    )


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 MCP Agent")
    st.markdown("---")
    st.markdown("### 🛠 Available Tools")
    st.markdown("**📧 send_email** — Gmail se email bhejo")
    st.markdown("**➕ add** — Do numbers jodo")
    st.markdown("**👋 greet** — Kisi ko greet karo")
    st.markdown("---")
    st.markdown("### 💡 Example Prompts")
    examples = [
        "Greet my friend Ali",
        "What is 450 + 320?",
        "Send an email to test@gmail.com with subject 'Hi' and body 'Hello!'",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["prefill"] = ex
            st.rerun()
    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    st.caption("Powered by LangGraph · Groq · yagmail")


# ─── Session State ───────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ─── Main ────────────────────────────────────────────────────────────────────
st.markdown("# 🤖 MCP Agent Chat")
st.markdown("Chat with an AI agent that can **send emails**, **do math**, and **greet people**.")
st.markdown("---")

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-msg">👤 <b>You</b><br>{msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="assistant-msg">🤖 <b>Agent</b><br>{msg["content"]}</div>',
            unsafe_allow_html=True
        )

prefill = st.session_state.pop("prefill", "")
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        "Message",
        value=prefill,
        placeholder="Type your message...",
        label_visibility="collapsed"
    )
with col2:
    send_clicked = st.button("Send →", use_container_width=True)

if send_clicked and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
    with st.spinner("Agent is thinking…"):
        try:
            agent = build_agent()
            response = agent.invoke({
                "messages": [{"role": "user", "content": user_input.strip()}]
            })
            messages = response.get("messages", [])
            reply = messages[-1].content if messages else "No response."
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"❌ Error: {str(e)}"
            })
    st.rerun()