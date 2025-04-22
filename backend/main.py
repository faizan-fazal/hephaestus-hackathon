import os
import aiohttp
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
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
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import AzureAISearchTool
from semantic_kernel.agents import AgentGroupChat, AzureAIAgent
from semantic_kernel.agents.strategies import DefaultTerminationStrategy

# Load environment variables
load_dotenv()

app = FastAPI()

# Credentials
APP_ID = os.getenv("MICROSOFT_APP_ID", "")
APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_AI_AGENT_PROJECT_CONNECTION_STRING = os.getenv("AZURE_AI_AGENT_PROJECT_CONNECTION_STRING")

# OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# Bot adapter
adapter_settings = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
adapter = BotFrameworkAdapter(adapter_settings)

# Conversation state
memory = MemoryStorage()
conversation_state = ConversationState(memory)

# Agent chat global (initialized later)
agent_chat: AgentGroupChat = None

# Helpers
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

# Initialize agents on startup
@app.on_event("startup")
async def initialize_agents():
    global agent_chat
    try:
        creds = DefaultAzureCredential()
        project_client = AIProjectClient.from_connection_string(
            conn_str=AZURE_AI_AGENT_PROJECT_CONNECTION_STRING,
            credential=creds
        )
        agent_client = AzureAIAgent.create_client(credential=creds)

        question_agent = project_client.agents.create_agent(
            model="gpt-4o-mini",
            name="question-agent",
            instructions="You are a thoughtful assistant that asks clarifying questions before answering."
        )
        answer_agent = project_client.agents.create_agent(
            model="gpt-4o-mini",
            name="answer-agent",
            instructions="You are a helpful AI that attempts to answer using uploaded document text if provided."
        )

        question_agent_definition = await agent_client.agents.get_agent(agent_id=question_agent.id)
        answer_agent_definition = await agent_client.agents.get_agent(agent_id=answer_agent.id)

        sk_question_agent = AzureAIAgent(client=agent_client, definition=question_agent_definition)
        sk_answer_agent = AzureAIAgent(client=agent_client, definition=answer_agent_definition)

        agent_chat = AgentGroupChat(
            agents=[sk_question_agent, sk_answer_agent],
            termination_strategy=DefaultTerminationStrategy(maximum_iterations=4)
        )
        print("✅ Agents initialized.")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")

# Message handler
async def on_message_activity(turn_context: TurnContext):
    prop = conversation_state.create_property("conversation_data")
    conversation_data = await prop.get(turn_context, lambda: {})

    # If document uploaded
    if turn_context.activity.attachments:
        attachment = turn_context.activity.attachments[0]
        file_url = attachment.content_url
        content_type = attachment.content_type
        try:
            text = await download_and_extract_text(file_url, content_type)
            conversation_data["last_uploaded_text"] = text
            await turn_context.send_activity("📄 Document received. You can now ask me what you want to do with it!")
        except Exception as e:
            await turn_context.send_activity(f"⚠️ Error processing file: {str(e)}")
        return

    # If user sends message
    elif turn_context.activity.type == "message" and turn_context.activity.text:
        user_input = turn_context.activity.text

        if agent_chat is None:
            await turn_context.send_activity("⚠️ Agent system not initialized.")
            return

        initial_prompt = user_input
        if "last_uploaded_text" in conversation_data:
            initial_prompt += f"\n\nHere is the document content:\n{conversation_data['last_uploaded_text']}"

        try:
            transcript = []
            async for msg in agent_chat.invoke(initial_message=initial_prompt):
                if msg:
                    transcript.append(f"{msg.name}: {msg.content}")
            final_response = "\n".join(transcript)
        except Exception as e:
            final_response = f"⚠️ Agent chat error: {str(e)}"

        await turn_context.send_activity(final_response)
        conversation_data.pop("last_uploaded_text", None)

    elif turn_context.activity.type == "conversationUpdate":
        for member in turn_context.activity.members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("👋 Hi! I'm HephAIstus — your AI assistant for this hackathon project. Upload a document or ask me anything to get started!")

    await conversation_state.save_changes(turn_context)

# Endpoint
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
