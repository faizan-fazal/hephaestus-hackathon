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
from docx import Document as DocxDocument

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

# Helpers
def extract_text_from_pdf(path):
    with open(path, "rb") as f:
        reader = PdfReader(f)
        return "\n".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_docx(path):
    doc = DocxDocument(path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

async def download_and_extract_text(url, content_type):
    suffix_map = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "text/plain": ".txt"
    }
    
    suffix = suffix_map.get(content_type)
    if not suffix:
        raise Exception("Unsupported file type")

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Download failed with status {resp.status}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await resp.read())
                tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            return extract_text_from_pdf(tmp_path)
        elif suffix == ".docx":
            return extract_text_from_docx(tmp_path)
        elif suffix == ".txt":
            return extract_text_from_txt(tmp_path)
    finally:
        os.remove(tmp_path)

# Message handler
async def on_message_activity(turn_context: TurnContext):
    conversation_data = await conversation_state.create_property("conversation_data").get(turn_context, {})

    # If document uploaded
    if turn_context.activity.attachments:
        attachment = turn_context.activity.attachments[0]
        file_url = attachment.content_url
        content_type = attachment.content_type

        try:
            text = await download_and_extract_text(file_url, content_type)
            conversation_data["last_uploaded_text"] = text
            await turn_context.send_activity("📄 Document received. You can now ask me to summarize it or analyze it!")
        except Exception as e:
            await turn_context.send_activity(f"⚠️ Error processing file: {str(e)}")
        return

    # If user sends message
    elif turn_context.activity.type == "message" and turn_context.activity.text:
        user_input = turn_context.activity.text
        history = [
            {"role": "system", "content": "You are an AI assistant helping users during a hackathon."},
        ]

        if "last_uploaded_text" in conversation_data:
            history.append({"role": "user", "content": f"Here is the document content:\n{conversation_data['last_uploaded_text']}"})
            history.append({"role": "user", "content": user_input})
        else:
            history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=history
            )
            ai_reply = response.choices[0].message.content
        except Exception as e:
            ai_reply = f"⚠️ Azure OpenAI error: {str(e)}"

        await turn_context.send_activity(ai_reply)
        conversation_data.pop("last_uploaded_text", None)

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
