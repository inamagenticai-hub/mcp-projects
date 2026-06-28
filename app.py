import streamlit as st
import asyncio
import json
from fastmcp import Client
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import BaseTool
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MCP Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    [data-testid="stSidebar"] {
        background: #0f1117;
        border-right: 1px solid #1e2130;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    .stApp { background: #0d0f18; }

    .user-msg {
        background: #1a1d2e;
        border-left: 3px solid #6366f1;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.95rem;
        color: #e2e8f0;
    }
    .assistant-msg {
        background: #111827;
        border-left: 3px solid #10b981;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.95rem;
        color: #d1fae5;
    }
    .tool-call-box {
        background: #1c1f2e;
        border: 1px solid #374151;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .tool-badge {
        display: inline-block;
        background: #312e81;
        color: #a5b4fc;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-bottom: 6px;
        font-family: 'JetBrains Mono', monospace;
    }
    .stTextInput > div > div > input {
        background: #1a1d2e !important;
        border: 1px solid #374151 !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.25) !important;
    }
    .stButton > button {
        background: #6366f1;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 22px;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #4f46e5; }
    .status-online  { color: #10b981; font-weight: 600; font-size: 0.8rem; }
    .status-offline { color: #ef4444; font-weight: 600; font-size: 0.8rem; }
    h1, h2, h3 { color: #f1f5f9 !important; }
    hr { border-color: #1e2130 !important; }
    [data-testid="stExpander"] {
        background: #1a1d2e;
        border: 1px solid #2d3148;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ─── MCP / Agent setup ──────────────────────────────────────────────────────

MCP_URL = "http://localhost:8000/mcp"
client = Client(MCP_URL)


class EmailInput(BaseModel):
    to: str
    subject: str
    body: str


class AddInput(BaseModel):
    a: int
    b: int


class GreetInput(BaseModel):
    name: str


async def call_mcp_tool(tool_name: str, **kwargs):
    async with client:
        return await client.call_tool(tool_name, kwargs)


async def get_all_tools():
    async with client:
        return await client.list_tools()


class MCPTool(BaseTool):
    name: str
    description: str
    mcp_tool_name: str

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs) -> str:
        result = asyncio.run(call_mcp_tool(self.mcp_tool_name, **kwargs))
        return str(result)

    async def _arun(self, tool_input: str) -> str:
        params = json.loads(tool_input)
        result = await call_mcp_tool(self.mcp_tool_name, **params)
        return str(result)


@st.cache_resource
def build_agent():
    tools = [
        MCPTool(
            name="send_email",
            description="Send a real email to any address. Requires: to (recipient email), subject (email subject), body (email text).",
            mcp_tool_name="send_email",
            args_schema=EmailInput,
        ),
        MCPTool(
            name="add",
            description="Add two integers. Requires: a (int), b (int).",
            mcp_tool_name="add",
            args_schema=AddInput,
        ),
        MCPTool(
            name="greet",
            description="Generate a greeting for a person. Requires: name (str).",
            mcp_tool_name="greet",
            args_schema=GreetInput,
        ),
    ]

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        system_prompt=(
            "You are a helpful assistant with access to the following tools:\n"
            "1. send_email — send real emails via Gmail\n"
            "2. add        — add two integers\n"
            "3. greet      — generate a greeting\n\n"
            "Use tools whenever the user's request calls for them. "
            "Be concise and accurate in your responses."
        ),
    )
    return agent, tools


def check_server_online() -> bool:
    try:
        asyncio.run(get_all_tools())
        return True
    except Exception:
        return False


def run_agent(agent, user_message: str, thread_id: str) -> dict:
    return agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config={"configurable": {"thread_id": thread_id}},
    )


# ─── Session state ───────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-session-1"
if "server_status" not in st.session_state:
    st.session_state.server_status = None


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 MCP Agent")
    st.markdown("---")

    if st.button("🔄 Check Server"):
        st.session_state.server_status = check_server_online()

    if st.session_state.server_status is True:
        st.markdown('<p class="status-online">● Server Online</p>', unsafe_allow_html=True)
    elif st.session_state.server_status is False:
        st.markdown('<p class="status-offline">● Server Offline</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#94a3b8;font-size:0.8rem;">○ Status unknown — click Check Server</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠 Available Tools")
    for tool_name, desc in {
        "📧 send_email": "Send a real email via Gmail",
        "➕ add": "Add two numbers together",
        "👋 greet": "Generate a greeting for someone",
    }.items():
        st.markdown(f"**{tool_name}**  \n{desc}")

    st.markdown("---")
    st.markdown("### ⚙️ Session")
    thread_id = st.text_input("Thread ID", value=st.session_state.thread_id)
    if thread_id != st.session_state.thread_id:
        st.session_state.thread_id = thread_id

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Example prompts")
    for ex in [
        "Send an email to ali@gmail.com with subject 'Meeting' and body 'See you at 3pm'",
        "What is 847 + 293?",
        "Greet my friend Sarah",
        "Send a leave request email to hr@company.com and also add 150 + 250",
    ]:
        if st.button(ex, key=ex):
            st.session_state["prefill"] = ex
            st.rerun()

    st.markdown("---")
    st.caption("Powered by FastMCP · LangGraph · Groq")


# ─── Main area ───────────────────────────────────────────────────────────────
st.markdown("# 🤖 MCP Agent Chat")
st.markdown("Chat with an AI agent that can **send emails**, **do math**, and **greet people** — all via MCP tools.")
st.markdown("---")

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-msg">👤 <b>You</b><br>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="assistant-msg">🤖 <b>Agent</b><br>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("tool_calls"):
            with st.expander("🔧 Tool calls used", expanded=False):
                for tc in msg["tool_calls"]:
                    st.markdown(
                        f'<div class="tool-call-box">'
                        f'<span class="tool-badge">TOOL: {tc["name"]}</span><br>'
                        f'{json.dumps(tc.get("args", {}), indent=2)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

prefill = st.session_state.pop("prefill", "")
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        "Message",
        value=prefill,
        placeholder="Type your message… e.g. 'Send an email to …' or 'Add 42 + 58'",
        label_visibility="collapsed",
        key="user_input_box",
    )
with col2:
    send_clicked = st.button("Send →", use_container_width=True)

if send_clicked and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})

    with st.spinner("Agent is thinking…"):
        try:
            agent, _ = build_agent()
            response = run_agent(agent, user_input.strip(), st.session_state.thread_id)

            messages = response.get("messages", [])
            final_content = messages[-1].content if messages else "No response."

            tool_calls_used = []
            for m in messages:
                if hasattr(m, "tool_calls") and m.tool_calls:
                    for tc in m.tool_calls:
                        tool_calls_used.append({
                            "name": tc.get("name", tc.get("function", {}).get("name", "unknown")),
                            "args": tc.get("args", tc.get("function", {}).get("arguments", {})),
                        })

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": final_content,
                "tool_calls": tool_calls_used,
            })

        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"❌ Error: {str(e)}\n\nMake sure the MCP server is running at `{MCP_URL}`.",
                "tool_calls": [],
            })

    st.rerun()