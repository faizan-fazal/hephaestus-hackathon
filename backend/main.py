import os
import aiohttp
import tempfile
from contextlib import asynccontextmanager
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
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import AzureAISearchTool
from semantic_kernel.agents import AgentGroupChat, AzureAIAgent
from semantic_kernel.agents.strategies import TerminationStrategy
from PyPDF2 import PdfReader
from docx import Document


# Lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    creds = DefaultAzureCredential()
    app.state.credential = creds
    app.state.project_client = AIProjectClient.from_connection_string(
        conn_str=os.environ["AZURE_AI_AGENT_PROJECT_CONNECTION_STRING"],
        credential=creds
    )
    app.state.agent_client = AzureAIAgent.create_client(credential=creds)
    yield


app = FastAPI(lifespan=lifespan)

# Credentials
APP_ID = os.getenv("MICROSOFT_APP_ID", "")
APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD", "")

# Bot adapter
adapter_settings = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
adapter = BotFrameworkAdapter(adapter_settings)

# Conversation state
memory = MemoryStorage()
conversation_state = ConversationState(memory)


# Helpers for file parsing
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


# Custom termination strategy
class ResponseTerminationStrategy(TerminationStrategy):
    async def should_agent_terminate(self, agent, history):
        return "FINAL RESPONSE" in history[-1].content


# Message handling
async def on_message_activity(turn_context: TurnContext):
    prop = conversation_state.create_property("conversation_data")
    conversation_data = await prop.get(turn_context, lambda: {})

    # File upload case
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

    # Message case
    elif turn_context.activity.type == "message" and turn_context.activity.text:
        user_input = turn_context.activity.text
        agent_client = app.state.agent_client
        project_client = app.state.project_client

        ai_search = AzureAISearchTool(
            index_connection_id=os.environ.get("SEARCH_ID"),
            index_name="pdf-index",
            query_type="vector_simple_hybrid"
        )

        answer_agent = project_client.agents.create_agent(
            model="gpt-4o-mini",
            name="answer-agent",
            instructions="You are a helpful assistant with access to Currie Bowman data.",
            tools=ai_search.definitions,
            tool_resources=ai_search.resources,
        )

        question_agent = project_client.agents.create_agent(
            model="gpt-4o-mini",
            name="question-agent",
            instructions="You are a geologist...",
        )

        sk_answer_agent = AzureAIAgent(client=agent_client, definition=await agent_client.agents.get_agent(answer_agent.id))
        sk_question_agent = AzureAIAgent(client=agent_client, definition=await agent_client.agents.get_agent(question_agent.id))

        chat = AgentGroupChat(
            agents=[sk_question_agent, sk_answer_agent],
            termination_strategy=ResponseTerminationStrategy()
        )

        await chat.add_chat_message(message=user_input)

        response_text = ""
        try:
            async for response in chat.invoke():
                if response and response.name:
                    response_text += f"\n\n🧠 {response.name} says:\n{response.content}"
        except Exception as e:
            response_text = f"⚠️ Agent chat error: {str(e)}"

        await turn_context.send_activity(response_text)

        # Clean up agents
        project_client.agents.delete_agent(answer_agent.id)
        project_client.agents.delete_agent(question_agent.id)

    elif turn_context.activity.type == "conversationUpdate":
        for member in turn_context.activity.members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("👋 Hi! I'm HephAIstus — your AI assistant for this hackathon. Upload a doc or ask a question!")

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
