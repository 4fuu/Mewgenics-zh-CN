import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def completion(messages: list, thinking_budget: int = 1024):
    completion = client.chat.completions.create(
        model="qwen3.5-plus",
        messages=messages,
        # temperature=0.5,
        extra_body={"enable_thinking": True, "thinking_budget": thinking_budget},
    )
    return completion.choices[0].message.content


def completion1(messages: list, thinking_budget: int = 0):
    completion = client.chat.completions.create(
        model="qwen3-max-2026-01-23",
        messages=messages,
        temperature=0.1,
        extra_body={
            "enable_thinking": True if thinking_budget != 0 else False,
            "thinking_budget": thinking_budget,
        },
    )
    return completion.choices[0].message.content
