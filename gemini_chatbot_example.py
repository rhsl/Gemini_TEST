import streamlit as st
import google.generativeai as genai
## 단일 메시지 채팅 샘플
with st.sidebar:
    gemini_api_key = st.text_input("Gemini API Key", key="chatbot_api_key", type="password")

st.title("💬 Google Gemini Chatbot")
st.caption("🚀 A streamlit chatbot powered by Google Gemini LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "어떻게 도와드릴까요?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not gemini_api_key:
        st.info("계속 진행하기 위해서는 당신의 Google Gemini API 키를 입력해주세요.")
        st.stop()

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    userMsg = st.session_state.messages[-1]['content']
    response = model.generate_content(userMsg)
    msg = response.text
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)