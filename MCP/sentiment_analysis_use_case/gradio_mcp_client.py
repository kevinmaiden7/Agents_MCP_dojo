"""Gradio MCP Client"""
from smolagents.mcp_client import MCPClient
import gradio as gr
import os
from smolagents import InferenceClientModel, CodeAgent
from dotenv import load_dotenv, find_dotenv

# Load the .env file
load_dotenv(find_dotenv())
# Retrieve HF_TOKEN from the environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

MCP_SERVER = "https://kevinmaiden7-mcp-sentiment.hf.space/gradio_api/mcp/sse"

def run_client() -> None:
    try:
        print(f"Establishing connection to {MCP_SERVER}")
        with MCPClient({"url": MCP_SERVER, "transport": "sse"}, structured_output=True) as tools:
            # Tools from the remote server are available
            print("Tools:")
            print("\n".join(f"{t.name}: {t.description}" for t in tools))

        # Gradio MCP Client

        mcp_client = MCPClient({"url": MCP_SERVER, "transport": "sse",}, structured_output=True)
        tools = mcp_client.get_tools()

        model = InferenceClientModel(token=HF_TOKEN)
        agent = CodeAgent(tools=[*tools], model=model, additional_authorized_imports=["json"])

        # Gradio Interface

        demo = gr.ChatInterface(
            fn=lambda message, history: str(agent.run(message)),
            type="messages",
            examples=["Analyze the sentiment of the following text 'This is awesome!'"],
            title="Agent with MCP Tools",
            description="This is a simple agent that uses MCP tools to do sentiment analysis.",
        )

        demo.launch()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        mcp_client.disconnect()

# Launch the interface
if __name__ == "__main__":
    run_client()
