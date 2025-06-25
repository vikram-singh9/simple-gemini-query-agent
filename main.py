from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig, set_tracing_disabled
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
    await cl.Message(content="Hello! How can I help you today?").send()

@cl.on_message
async def on_message(message: cl.Message):
    #saving user message to session history
    history = cl.user_session.get('history',[])
    history.append({
        'role': 'user',
        'content': message.content
    })

    # Run the agent using runner
    result = await Runner.run(agent, input=history)

    # Append the agent's response to the session history
    history.append({
        'role': 'assistant',
        'content': result.final_output
    })
    cl.user_session.set('history', history)

    # Send back the agent's final output to UI
    cl.Message(content="Processing your request...").send()
    await cl.Message(content=result.final_output).send()