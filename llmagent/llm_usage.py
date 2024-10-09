import os
import openai

# unset all_proxy and ALL_PROXY
os.environ['all_proxy'] = ""
os.environ['ALL_PROXY'] = ""

# set api_key and base_url here
api_key = "sk-RRKTiQPZGTlk25SHFb3d44F4E73146519086F3F6B3E178F1"
base_url = "https://free.gpt.ge/v1/"
default_headers = {"x-foo": "true"}

# initialize prompt
messages=[
    {
        "role": "system",
        "content": "You are a helpful assistent. Your name is robocasa",
    },
    {
        "role": "user",
        "content": "Hello world! Who are you? Please output single word",
    },
]

# usage 1
# optional; defaults to `os.environ['OPENAI_API_KEY']`
openai.api_key = api_key
openai.base_url = base_url
openai.default_headers = default_headers

completion1 = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
print("answer1: ", completion1.choices[0].message.content)

# usage 2
client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url,
    default_headers=default_headers
)

completion2 = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
)
print("answer2: ", completion2.choices[0].message.content)