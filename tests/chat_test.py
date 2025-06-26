from openai import OpenAI

client = OpenAI()

history = [
    {
        "role": "user",
        "content": "tell me a joke"
    }
]

response = client.responses.create(
    model="gpt-4.1-nano",
    input=history,
    store=False
)

history += [{"role": el.role, "content": el.content} for el in response.output]
print(history[1]['content'][0])
print(type(history[1]['content'][0]))
exit()
history.append({ "role": "user", "content": "tell me another" })

second_response = client.responses.create(
    model="gpt-4.1-nano",
    input=history,
    store=False
)
print(second_response.output)