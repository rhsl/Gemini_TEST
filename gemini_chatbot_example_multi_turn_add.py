# Reference : 
# 1. Google Gemini API - https://ai.google.dev/tutorials/python_quickstart
# 2. OpenAI API - https://platform.openai.com/docs/guides/text-generation/chat-completions-api

# 1. OpenAI API message example
# messages=[
#     {"role": "user", "content": "Who won the world series in 2020?"},
#     {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#     {"role": "user", "content": "Where was it played?"}
# ]

# 2. Google Gemini API (Multi-trun) message example
# messages = [
#     {'role':'user',
#      'parts': ["Briefly explain how a computer works to a young child."]},
#     {'role':'model', 'parts':[response.text]},
#     {'role':'user', 'parts':["Okay, how about a more detailed explanation to a high school student?"]}
# ]


import streamlit as st
import google.generativeai as genai

with st.sidebar:
    gemini_api_key = st.text_input("Gemini API Key", key="chatbot_api_key", type="password")

st.title("💬 Google Gemini Chatbot (Multi-turn 기능 추가)")
st.caption("🚀 A streamlit chatbot powered by Google Gemini LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["parts"][-1])

if prompt := st.chat_input():
    if not gemini_api_key:
        st.info("계속 진행하기 위해서는 당신의 Google Gemini API 키를 입력해주세요.")
        st.stop()

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')

    st.session_state.messages.append({"role": "user", "parts": [prompt]})
    st.chat_message("user").write(prompt)
    response = model.generate_content(st.session_state.messages)
    msg = response.text
    st.session_state.messages.append({"role": "model", "parts": [msg]})
    st.chat_message("model").write(msg)