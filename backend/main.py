import os
import aiohttp
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from contextlib import asynccontextmanager
from azure.identity import ClientSecretCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import AzureAISearchTool
from semantic_kernel.agents import AgentGroupChat, AzureAIAgent
from semantic_kernel.agents.strategies import TerminationStrategy
from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext,
    MemoryStorage,
    ConversationState
)
from botbuilder.schema import Activity
from openai import AzureOpenAI
from PyPDF2 import PdfReader
from docx import Document

# Load environment variables
load_dotenv()

# FastAPI lifespan for agent setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global sk_answer_agent, sk_question_agent, chat

    creds = ClientSecretCredential(
        tenant_id=os.getenv("AZURE_TENANT_ID"),
        client_id=os.getenv("AZURE_CLIENT_ID"),
        client_secret=os.getenv("AZURE_CLIENT_SECRET")
    )

    agent_client = AzureAIAgent.create_client(credential=creds)
    project_client = AIProjectClient.from_connection_string(
        conn_str=os.getenv("AZURE_AI_AGENT_PROJECT_CONNECTION_STRING"),
        credential=creds
    )

    ai_search = AzureAISearchTool(
        index_connection_id=os.getenv("SEARCH_ID"),
        index_name="pdf-index",
        query_type="vector_simple_hybrid"
    )

    answer_agent = project_client.agents.create_agent(
        model=os.getenv("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"),
        name="answer-agent",
        instructions="You are a helpful assistant with access to data about Currie Bowman property.",
        tools=ai_search.definitions,
        tool_resources=ai_search.resources,
    )

    question_agent = project_client.agents.create_agent(
        model=os.getenv("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME"),
        name="question-agent",
        instructions="You are a geologist collecting information for comprehensive property analysis."
    )

    answer_agent_definition = await agent_client.agents.get_agent(agent_id=answer_agent.id)
    question_agent_definition = await agent_client.agents.get_agent(agent_id=question_agent.id)

    sk_answer_agent = AzureAIAgent(client=agent_client, definition=answer_agent_definition)
    sk_question_agent = AzureAIAgent(client=agent_client, definition=question_agent_definition)

    class ResponseTerminationStrategy(TerminationStrategy):
        async def should_agent_terminate(self, agent, history):
            return "FINAL RESPONSE" in history[-1].content

    chat = AgentGroupChat(
        agents=[sk_question_agent, sk_answer_agent],
        termination_strategy=ResponseTerminationStrategy(maximum_iterations=10)
    )

    yield

# App Initialization
app = FastAPI(lifespan=lifespan)

# OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-12-01-preview"
)

# Bot adapter
adapter_settings = BotFrameworkAdapterSettings(
    os.getenv("MICROSOFT_APP_ID", ""),
    os.getenv("MICROSOFT_APP_PASSWORD", "")
)
adapter = BotFrameworkAdapter(adapter_settings)
memory = MemoryStorage()
conversation_state = ConversationState(memory)

# Document extraction

def extract_text_from_pdf(path):
    with open(path, "rb") as f:
        reader = PdfReader(f)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

async def download_and_extract_text(url, content_type):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Download failed with status {resp.status}")
            suffix = {
                "application/pdf": ".pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                "text/plain": ".txt"
            }.get(content_type, None)
            if suffix is None:
                raise Exception("Unsupported file type")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await resp.read())
                tmp_path = tmp.name

    if suffix == ".pdf":
        text = extract_text_from_pdf(tmp_path)
    elif suffix == ".docx":
        text = extract_text_from_docx(tmp_path)
    elif suffix == ".txt":
        text = extract_text_from_txt(tmp_path)

    os.remove(tmp_path)
    return text

# Message handler
async def on_message_activity(turn_context: TurnContext):
    prop = conversation_state.create_property("conversation_data")
    conversation_data = await prop.get(turn_context, lambda: {})

    if turn_context.activity.attachments:
        attachment = turn_context.activity.attachments[0]
        try:
            text = await download_and_extract_text(attachment.content_url, attachment.content_type)
            conversation_data["last_uploaded_text"] = text
            await turn_context.send_activity("\ud83d\udcc4 Document received. You can now ask me what you want to do with it!")
        except Exception as e:
            await turn_context.send_activity(f"\u26a0\ufe0f Error processing file: {str(e)}")
        return

    elif turn_context.activity.type == "message" and turn_context.activity.text:
        user_input = turn_context.activity.text
        await chat.add_chat_message(message=user_input)
        try:
            async for response in chat.invoke():
                if response and response.name:
                    await turn_context.send_activity(f"#{response.name.upper()}:\n{response.content}")
        except Exception as e:
            await turn_context.send_activity(f"\u26a0\ufe0f Chat error: {str(e)}")

    elif turn_context.activity.type == "conversationUpdate":
        for member in turn_context.activity.members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(
                    "\ud83d\udc4b Hi! I'm HephAIstus — your AI assistant for this hackathon project. Upload a document or ask me anything to get started!"
                )

    await conversation_state.save_changes(turn_context)

# API endpoint
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
        print("\u274c EXCEPTION:", str(e))
        return Response(content=f"Error: {e}", status_code=500)
