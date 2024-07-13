from litellm import completion, embedding


def llm_call(messages, model="gpt-3.5-turbo"):
    response = completion(model=model, messages=messages)
    return response.choices[0].message.content


def llm_call_multiple_choices(messages, model="gpt-3.5-turbo", n=1, temperature=0):
    response = completion(model=model, messages=messages, n=n, temperature=temperature)
    return [choice.message.content for choice in response.choices]


def get_embedding(input_text, model="text-embedding-ada-002"):
    response = embedding(model=model, input=[input_text])
    return response.data[0]["embedding"]
