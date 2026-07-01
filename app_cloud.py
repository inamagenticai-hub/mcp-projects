import streamlit as st
import yagmail
import os
import math
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


# ─── Math Tools ─────────────────────────────────────────────────────────────
@tool
def add(input: str) -> str:
    """ALWAYS use this tool to add two numbers. Input format: 'a,b' e.g. '70,10'"""
    try:
        parts = input.replace(" ", "").split(",")
        a, b = float(parts[0]), float(parts[1])
        return f"{a} + {b} = {a + b}"
    except:
        return "❌ Please provide input as 'a,b' e.g. '70,10'"

@tool
def subtract(input: str) -> str:
    """ALWAYS use this tool to subtract two numbers. Input format: 'a,b' e.g. '50,10'"""
    try:
        parts = input.replace(" ", "").split(",")
        a, b = float(parts[0]), float(parts[1])
        return f"{a} - {b} = {a - b}"
    except:
        return "❌ Please provide input as 'a,b' e.g. '50,10'"

@tool
def multiply(input: str) -> str:
    """ALWAYS use this tool to multiply two numbers. Input format: 'a,b' e.g. '6,7'"""
    try:
        parts = input.replace(" ", "").split(",")
        a, b = float(parts[0]), float(parts[1])
        return f"{a} × {b} = {a * b}"
    except:
        return "❌ Please provide input as 'a,b' e.g. '6,7'"

@tool
def divide(input: str) -> str:
    """ALWAYS use this tool to divide two numbers. Input format: 'a,b' e.g. '100,4'"""
    try:
        parts = input.replace(" ", "").split(",")
        a, b = float(parts[0]), float(parts[1])
        if b == 0:
            return "❌ Cannot divide by zero."
        return f"{a} ÷ {b} = {a / b}"
    except:
        return "❌ Please provide input as 'a,b' e.g. '100,4'"

@tool
def percentage(input: str) -> str:
    """ALWAYS use this tool to find what percentage value is of total. Input format: 'value,total' e.g. '25,200'"""
    try:
        parts = input.replace(" ", "").split(",")
        value, total = float(parts[0]), float(parts[1])
        if total == 0:
            return "❌ Total cannot be zero."
        result = (value / total) * 100
        return f"{value} is {result:.2f}% of {total}"
    except:
        return "❌ Please provide input as 'value,total' e.g. '25,200'"

@tool
def percentage_of(input: str) -> str:
    """ALWAYS use this tool when user asks what X percent of Y is. Input format: 'percent,total' e.g. '15,2000'"""
    try:
        parts = input.replace(" ", "").split(",")
        percent, total = float(parts[0]), float(parts[1])
        result = (percent / 100) * total
        return f"{percent}% of {total} = {result}"
    except:
        return "❌ Please provide input as 'percent,total' e.g. '15,2000'"

@tool
def power(input: str) -> str:
    """ALWAYS use this tool to calculate power or exponent. Input format: 'base,exponent' e.g. '2,10'"""
    try:
        parts = input.replace(" ", "").split(",")
        base, exp = float(parts[0]), float(parts[1])
        return f"{base}^{exp} = {base ** exp}"
    except:
        return "❌ Please provide input as 'base,exponent' e.g. '2,10'"

@tool
def square_root(input: str) -> str:
    """ALWAYS use this tool to calculate square root of a number. Input format: 'number' e.g. '144'"""
    try:
        number = float(input.strip())
        if number < 0:
            return "❌ Cannot calculate square root of a negative number."
        return f"√{number} = {math.sqrt(number)}"
    except:
        return "❌ Please provide a valid number e.g. '144'"

@tool
def factorial(input: str) -> str:
    """ALWAYS use this tool to calculate factorial of a number. Input format: 'number' e.g. '5'"""
    try:
        n = int(float(input.strip()))
        if n < 0:
            return "❌ Factorial of negative number is not defined."
        if n > 20:
            return "❌ Number too large, please use a number <= 20."
        return f"{n}! = {math.factorial(n)}"
    except:
        return "❌ Please provide a valid number e.g. '5'"

@tool
def modulus(input: str) -> str:
    """ALWAYS use this tool to find remainder after division. Input format: 'a,b' e.g. '17,5'"""
    try:
        parts = input.replace(" ", "").split(",")
        a, b = float(parts[0]), float(parts[1])
        if b == 0:
            return "❌ Cannot divide by zero."
        return f"{a} mod {b} = {a % b}"
    except:
        return "❌ Please provide input as 'a,b' e.g. '17,5'"

@tool
def absolute_value(input: str) -> str:
    """ALWAYS use this tool to find absolute value of a number. Input format: 'number' e.g. '-45'"""
    try:
        number = float(input.strip())
        return f"|{number}| = {abs(number)}"
    except:
        return "❌ Please provide a valid number e.g. '-45'"

@tool
def average(input: str) -> str:
    """ALWAYS use this tool to find average of numbers. Input format: 'a,b,c,...' e.g. '10,20,30,40'"""
    try:
        nums = [float(x.strip()) for x in input.split(",")]
        avg = sum(nums) / len(nums)
        return f"Average = {avg:.2f}"
    except:
        return "❌ Please provide numbers separated by commas e.g. '10,20,30'"

@tool
def log(input: str) -> str:
    """ALWAYS use this tool to find logarithm. Input format: 'number' or 'number,base' e.g. '1000' or '8,2'"""
    try:
        parts = input.replace(" ", "").split(",")
        number = float(parts[0])
        if number <= 0:
            return "❌ Logarithm is only defined for positive numbers."
        if len(parts) == 2:
            base = float(parts[1])
            return f"log base {base} of {number} = {math.log(number, base):.4f}"
        return f"log({number}) = {math.log10(number):.4f}"
    except:
        return "❌ Please provide input as 'number' or 'number,base'"
    
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
            "You are a helpful assistant with access to math and utility tools.\n"
            "IMPORTANT RULES:\n"
            "1. ALWAYS use the correct tool for every math operation.\n"
            "2. ALWAYS use 'greet' tool when user wants to greet someone.\n"
            "3. ALWAYS use 'send_email' tool when user wants to send an email.\n"
            "4. NEVER calculate math yourself — always use the tool.\n"
            "5. Show the exact output returned by the tool in your response.\n"
            "6. For average, pass numbers as comma separated string e.g. '10,20,30'."
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
        "What is 125 + 378?",
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
if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0


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