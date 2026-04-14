from openai import OpenAI

client = OpenAI(api_key="")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Reply with the word hello only."}],
    max_tokens=10,
)

print(response.choices[0].message.content)