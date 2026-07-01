import streamlit as st
import yagmail
import os
import math
from langchain_groq import ChatGroq
from langchain.tools import StructuredTool
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


# ─── Tool Functions ──────────────────────────────────────────────────────────
def _add(a: float, b: float) -> str:
    return f"{a} + {b} = {a + b}"

def _subtract(a: float, b: float) -> str:
    return f"{a} - {b} = {a - b}"

def _multiply(a: float, b: float) -> str:
    return f"{a} × {b} = {a * b}"

def _divide(a: float, b: float) -> str:
    if b == 0:
        return "❌ Cannot divide by zero."
    return f"{a} ÷ {b} = {a / b}"

def _percentage(value: float, total: float) -> str:
    if total == 0:
        return "❌ Total cannot be zero."
    return f"{value} is {(value/total)*100:.2f}% of {total}"

def _percentage_of(percent: float, total: float) -> str:
    return f"{percent}% of {total} = {(percent/100)*total}"

def _power(base: float, exponent: float) -> str:
    return f"{base}^{exponent} = {base ** exponent}"

def _square_root(number: float) -> str:
    if number < 0:
        return "❌ Cannot calculate square root of negative number."
    return f"√{number} = {math.sqrt(number)}"

def _factorial(number: int) -> str:
    if number < 0:
        return "❌ Factorial of negative number is not defined."
    if number > 20:
        return "❌ Number too large, use number <= 20."
    return f"{number}! = {math.factorial(number)}"

def _modulus(a: float, b: float) -> str:
    if b == 0:
        return "❌ Cannot divide by zero."
    return f"{a} mod {b} = {a % b}"

def _absolute_value(number: float) -> str:
    return f"|{number}| = {abs(number)}"

def _average(numbers: str) -> str:
    try:
        nums = [float(x.strip()) for x in numbers.split(",")]
        return f"Average = {sum(nums)/len(nums):.2f}"
    except:
        return "❌ Provide numbers separated by commas e.g. '10,20,30'"

def _log(number: float, base: float = 10.0) -> str:
    if number <= 0:
        return "❌ Logarithm only defined for positive numbers."
    if base == 10:
        return f"log({number}) = {math.log10(number):.4f}"
    return f"log base {base} of {number} = {math.log(number, base):.4f}"

def _greet(name: str) -> str:
    return f"Hello, {name}! Hope you are doing well! 😊"

def _send_email(to: str, subject: str, body: str) -> str:
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


# ─── StructuredTools ─────────────────────────────────────────────────────────
tools = [
    StructuredTool.from_function(_add,        name="add",            description="Add two numbers. Args: a (number), b (number)"),
    StructuredTool.from_function(_subtract,   name="subtract",       description="Subtract two numbers. Args: a (number), b (number)"),
    StructuredTool.from_function(_multiply,   name="multiply",       description="Multiply two numbers. Args: a (number), b (number)"),
    StructuredTool.from_function(_divide,     name="divide",         description="Divide two numbers. Args: a (number), b (number)"),
    StructuredTool.from_function(_percentage, name="percentage",     description="Find what percentage value is of total. Args: value (number), total (number)"),
    StructuredTool.from_function(_percentage_of, name="percentage_of", description="Find X percent of Y. Args: percent (number), total (number)"),
    StructuredTool.from_function(_power,      name="power",          description="Calculate base raised to exponent. Args: base (number), exponent (number)"),
    StructuredTool.from_function(_square_root,name="square_root",    description="Calculate square root of a number. Args: number (number)"),
    StructuredTool.from_function(_factorial,  name="factorial",      description="Calculate factorial of a number. Args: number (int)"),
    StructuredTool.from_function(_modulus,    name="modulus",        description="Find remainder after division. Args: a (number), b (number)"),
    StructuredTool.from_function(_absolute_value, name="absolute_value", description="Find absolute value of a number. Args: number (number)"),
    StructuredTool.from_function(_average,    name="average",        description="Find average of numbers. Args: numbers (comma separated string e.g. '10,20,30')"),
    StructuredTool.from_function(_log,        name="log",            description="Find logarithm of a number. Args: number (number), base (number, default 10)"),
    StructuredTool.from_function(_greet,      name="greet",          description="Greet a person by name. Args: name (string)"),
    StructuredTool.from_function(_send_email, name="send_email",     description="Send an email via Gmail. Args: to (email), subject (string), body (string)"),
]


# ─── Agent ──────────────────────────────────────────────────────────────────
@st.cache_resource
def build_agent():
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        max_tokens=1024,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=(
            "You are a helpful assistant with math and utility tools.\n"
            "RULES:\n"
            "1. ALWAYS use the correct tool for every operation.\n"
            "2. NEVER calculate math yourself — always use the tool.\n"
            "3. Show the exact output returned by the tool.\n"
            "4. For average, pass numbers as comma separated string e.g. '10,20,30'."
        )
    )


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 MCP Agent")
    st.markdown("---")
    st.markdown("### 🛠 Available Tools")
    st.markdown("**📧 send_email** — Send emails via Gmail")
    st.markdown("**👋 greet** — Generate a greeting for someone")
    st.markdown("---")
    st.markdown("**➕ add** — Add two numbers")
    st.markdown("**➖ subtract** — Subtract two numbers")
    st.markdown("**✖️ multiply** — Multiply two numbers")
    st.markdown("**➗ divide** — Divide two numbers")
    st.markdown("**📊 percentage** — Calculate percentage")
    st.markdown("**📊 percentage_of** — Find X% of Y")
    st.markdown("**🔢 power** — Calculate power/exponent")
    st.markdown("**√ square_root** — Square root of a number")
    st.markdown("**❗ factorial** — Factorial of a number")
    st.markdown("**🔁 modulus** — Remainder after division")
    st.markdown("**🔢 absolute_value** — Absolute value")
    st.markdown("**📈 average** — Average of numbers")
    st.markdown("**📉 log** — Logarithm of a number")
    st.markdown("---")
    st.markdown("### 💡 Example Prompts")
    examples = [
        "What is 70 + 20?",
        "What is 50 - 10?",
        "Multiply 45 by 13",
        "What is square root of 144?",
        "Calculate average of 10, 20, 30, 40, 50",
        "What is 15% of 2000?",
        "Greet my friend Ali",
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
        st.markdown(f'<div class="user-msg">👤 <b>You</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-msg">🤖 <b>Agent</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

prefill = st.session_state.pop("prefill", "")

with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Message",
            value=prefill,
            placeholder="Type your message...",
            label_visibility="collapsed",
        )
    with col2:
        send_clicked = st.form_submit_button("Send →", use_container_width=True)

if send_clicked and user_input.strip():
    msg = user_input.strip()
    st.session_state.chat_history.append({"role": "user", "content": msg})
    with st.spinner("Agent is thinking…"):
        try:
            agent = build_agent()
            response = agent.invoke({
                "messages": [{"role": "user", "content": msg}]
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