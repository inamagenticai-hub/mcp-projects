from fastmcp import FastMCP
import yagmail
import os
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("General Utility Server")

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Returns the sum of two integers."""
    return a + b


@mcp.tool()
def greet(name: str) -> str:
    """Returns a greeting message for the given name."""
    return f"Hello, {name}!"


@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """
    Sends a real email using yagmail (Gmail SMTP).
    Args:
        to:      Recipient email address
        subject: Email subject line
        body:    Email body (plain text)
    Returns:
        Confirmation string or error message
    """
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        return "Email credentials missing. Please set SENDER_EMAIL and SENDER_PASSWORD in your .env file."

    try:
        yag = yagmail.SMTP(user=SENDER_EMAIL, password=SENDER_PASSWORD)
        yag.send(to=to, subject=subject, contents=body)
        return f"✅ Email successfully sent to {to} | Subject: '{subject}'"
    except yagmail.error.YagAddressError:
        return f"❌ Invalid recipient address: {to}"
    except yagmail.error.YagConnectionClosed as e:
        return f"❌ Connection error: {e}"
    except Exception as e:
        return f"❌ Failed to send email: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="http", host="localhost", port=8000)