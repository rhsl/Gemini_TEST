from openai import OpenAI
import streamlit as st
import requests
from PIL import Image

with st.sidebar:    
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

st.title("💬 OpenAI DALL·E Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI API")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    if msg["role"] == 'user':
        st.chat_message(msg["role"]).write(msg["content"])
    elif msg["role"] == 'assistant':
        st.image(
            Image.open(requests.get(msg["content"][0], stream=True).raw),
            caption=f'{msg["content"][1]}',
            use_column_width=True,
        )        

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = OpenAI(api_key=openai_api_key)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response = client.images.generate(
        model="dall-e-3",
        prompt=f"{prompt}",
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    revised_prompt = response.data[0].revised_prompt

    st.image(
        Image.open(requests.get(image_url, stream=True).raw),
        caption=f"{revised_prompt}",
        use_column_width=True,
    )

    st.session_state.messages.append({"role": "assistant", "content": [image_url, revised_prompt]})
