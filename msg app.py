import os
import gradio as gr
from groq import Groq

# Fetch API key from environment variable
api_key = "write_your_own_api_key_here"

client = Groq(api_key=api_key)

def chatbot(message, history):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        stream=True,
    )

    partial_message = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            yield partial_message

iface = gr.ChatInterface(
    chatbot,
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="Type your message here...", container=False, scale=7),
    title="KARTIK's BOT",
    description="Chat with kartik 6.12b LLM ",
    theme="soft",
    examples=[
        "What is Data Science?",
        "Can you explain Stock market?",
        "Tell me a joke about programming.",
    ],
    cache_examples=True
)

if __name__ == "__main__":
    iface.launch(share=True)
