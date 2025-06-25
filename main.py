from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
from openai import AsyncOpenAI
import os
import chainlit as cl

load_dotenv()

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
    instructions='You are a helpful assistant.',
    model=model
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)


@cl.on_chat_start
async def handle_chat_start():
    await cl.Message(content="Hello! How can I help you today?").send()

@cl.on_message
async def on_message(message: cl.Message):
    # Receive user input from Chainlit frontend
    user_input = message.content

    # Run the agent using runner
    result = await Runner.run(agent, input=user_input)

    # Send back the agent's final output to UI
    await cl.Message(content=result.final_output).send()