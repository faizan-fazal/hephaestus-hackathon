import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
    TurnContext
)
from botbuilder.schema import Activity
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

# Microsoft Bot credentials
APP_ID = os.getenv("MICROSOFT_APP_ID", "")
APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD", "")

# Azure OpenAI credentials
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-10-21",  
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Bot Adapter setup
adapter_settings = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
adapter = BotFrameworkAdapter(adapter_settings)

# Handle incoming message activity
async def on_message_activity(turn_context: TurnContext):
    if turn_context.activity.type == "message" and turn_context.activity.text:
        user_input = turn_context.activity.text

        try:
            # New SDK call
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are an AI assistant helping users during a hackathon."},
                    {"role": "user", "content": user_input}
                ]
            )
            ai_reply = response.choices[0].message.content
        except Exception as e:
            ai_reply = f"⚠️ Azure OpenAI error: {str(e)}"

        await turn_context.send_activity(ai_reply)

    elif turn_context.activity.type == "conversationUpdate":
        for member in turn_context.activity.members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(
                    "👋 Hi! I'm HephAIstus — your AI assistant for this hackathon project. Ask me anything!"
                )

# Bot message handler endpoint
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
