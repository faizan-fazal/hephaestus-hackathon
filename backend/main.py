import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext
)
from botbuilder.schema import Activity
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import AzureAISearchTool
from semantic_kernel.agents import AgentGroupChat, AzureAIAgent
from semantic_kernel.agents.strategies import DefaultTerminationStrategy

# Load environment variables
load_dotenv()

# ---------------------------
# FASTAPI APP & BOT CONFIGURATION
# ---------------------------
app = FastAPI()
APP_ID = os.getenv("MICROSOFT_APP_ID", "")
APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD", "")

AZURE_PROJECT_ID = os.getenv("AZURE_AI_PROJECT_ID")
AZURE_PROJECT_ENDPOINT = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
SEARCH_ID = os.getenv("SEARCH_ID")  # Optional

adapter_settings = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
adapter = BotFrameworkAdapter(adapter_settings)

# Global variable to hold chat
chat: AgentGroupChat = None

# ---------------------------
# ASYNC AGENT INITIALIZATION
# ---------------------------
async def initialize_agents():
    creds = DefaultAzureCredential()

    print(f"🔍 Connecting to Azure project: {AZURE_PROJECT_ID}")
    project_client = AIProjectClient(
        endpoint=AZURE_PROJECT_ENDPOINT,
        project_id=AZURE_PROJECT_ID,
        credential=creds
    )

    agent_client = AzureAIAgent.create_client(credential=creds)

    ai_search = AzureAISearchTool(
        index_connection_id=SEARCH_ID,
        index_name="pdf-index",
        query_type="vector_simple_hybrid"
    )

    question_agent = project_client.agents.create_agent(
        model="gpt-4o-mini",
        name="question-agent",
        instructions="You are a geologist who asks specific questions about the Currie Bowman property."
    )
    answer_agent = project_client.agents.create_agent(
        model="gpt-4o-mini",
        name="answer-agent",
        instructions="You are a helpful assistant with access to data about the Currie Bowman property.",
        tools=ai_search.definitions,
        tool_resources=ai_search.resources,
    )

    question_agent_definition = await agent_client.agents.get_agent(agent_id=question_agent.id)
    answer_agent_definition = await agent_client.agents.get_agent(agent_id=answer_agent.id)

    sk_question_agent = AzureAIAgent(client=agent_client, definition=question_agent_definition)
    sk_answer_agent = AzureAIAgent(client=agent_client, definition=answer_agent_definition)

    return AgentGroupChat(
        agents=[sk_question_agent, sk_answer_agent],
        termination_strategy=DefaultTerminationStrategy(maximum_iterations=4)
    )

# ---------------------------
# STARTUP EVENT
# ---------------------------
@app.on_event("startup")
async def startup_event():
    global chat
    print("🚀 Startup event triggered. Initializing agents...")

    try:
        chat = await initialize_agents()
        print("✅ AgentGroupChat initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agents: {e}")
        chat = None

# ---------------------------
# BOT ACTIVITY HANDLING
# ---------------------------
async def on_message_activity(turn_context: TurnContext):
    global chat

    if turn_context.activity.type == "message" and turn_context.activity.text:
        user_input = turn_context.activity.text
        transcript_lines = []

        if chat is None:
            await turn_context.send_activity("⚠️ Sorry, the AI assistant failed to start. Please try again later.")
            return

        try:
            async for response in chat.invoke(initial_message=user_input):
                if response is None or not response.name:
                    continue
                transcript_lines.append(f"{response.name}: {response.content}")
            ai_reply = "\n".join(transcript_lines)
        except Exception as e:
            ai_reply = f"⚠️ Agent error: {str(e)}"

        await turn_context.send_activity(ai_reply)

    elif turn_context.activity.type == "conversationUpdate":
        for member in turn_context.activity.members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(
                    "👋 Hi! I'm HephAIstus — your AI assistant for this hackathon project. Ask me anything!"
                )

# ---------------------------
# FASTAPI ENDPOINT
# ---------------------------
@app.post("/api/messages")
async def messages(req: Request):
    try:
        json_body = await req.json()
        activity = Activity().deserialize(json_body)
        auth_header = req.headers.get("Authorization", "")

        async def aux_func(turn_context: TurnContext):
            await on_message_activity(turn_context)

        await adapter.process_activity(activity, auth_header, aux_func)
        return Response(status_code=200)

    except Exception as e:
        print("❌ EXCEPTION:", str(e))
        return Response(content=f"Error: {e}", status_code=500)
