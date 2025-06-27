from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig, set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os
import chainlit as cl

load_dotenv()
set_tracing_disabled(True)

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

model = OpenAIChatCompletionsModel(
        model = 'gemini-2.0-flash',
        openai_client=client,
    )

agent = Agent(
    name='assistant',
    instructions='your are a query agent.',
    model=model
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)


@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set('history',[])
    await cl.Message(content="It's VIKRAM! How can I help you today?").send()

@cl.on_message
async def on_message(message: cl.Message):
    #saving user message to session history
    history = cl.user_session.get('history',[])
    history.append({
        'role': 'user',
        'content': message.content
    })

    msg = cl.Message(content="")
    msg.send()

    # Run the agent using runner
    result = Runner.run_streamed(agent, input=history)

    async for event in result.stream_events():
        if event.type == 'raw_response_event' and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
            
    # Append the agent's response to the session history
    history.append({
        'role': 'assistant',
        'content': result.final_output
    })
    cl.user_session.set('history', history)

    # Send back the agent's final output to UI
    # await cl.Message(content=result.final_output).send() we are running through streaming