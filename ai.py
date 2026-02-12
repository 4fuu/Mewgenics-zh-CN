import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def completion(messages: list):
    completion = client.chat.completions.create(
        model="qwen3-max-2026-01-23", messages=messages, temperature=0.1
    )
    return completion.choices[0].message.content
