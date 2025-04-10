import os
import asyncio
import tempfile
import requests
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
AZURE_CONNECTION_STRING = os.getenv("AZURE_AI_AGENT_PROJECT_CONNECTION_STRING")
SEARCH_ID = os.getenv("SEARCH_ID")  # Optional

adapter_settings = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
adapter = BotFrameworkAdapter(adapter_settings)

# Globals to hold agent chat session and uploaded document text
chat: AgentGroupChat = None
user_documents = {}

# ---------------------------
# ASYNC AGENT INITIALIZATION
# ---------------------------
async def initialize_agents():
    creds = DefaultAzureCredential()
    project_client = AIProjectClient.from_connection_string(
        conn_str=AZURE_CONNECTION_STRING,
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
        instructions="You are a helpful assistant with access to data about Currie Bowman property.",
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

@app.on_event("startup")
async def startup_event():
    global chat
    chat = await initialize_agents()

# ---------------------------
# BOT ACTIVITY HANDLING
# ---------------------------
async def on_message_activity(turn_context: TurnContext):
    user_id = turn_context.activity.from_property.id

    if turn_context.activity.attachments:
        attachment = turn_context.activity.attachments[0]
        content_url = attachment.content_url
        headers = {"Authorization": f"Bearer {turn_context.activity.service_url}"}
        response = requests.get(content_url, headers=headers)

        if response.status_code == 200:
            text_content = response.content.decode("utf-8", errors="ignore")
            user_documents[user_id] = text_content
            await turn_context.send_activity("📎 Got your file. Now tell me what you’d like me to do with it.")
        else:
            await turn_context.send_activity("⚠️ Failed to retrieve the file.")

    elif turn_context.activity.type == "message" and turn_context.activity.text:
        user_input = turn_context.activity.text
        transcript_lines = []

        try:
            if user_id in user_documents:
                user_input = f"Please summarize the following document:\n{user_documents[user_id]}\n\nInstruction: {user_input}"
                del user_documents[user_id]

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