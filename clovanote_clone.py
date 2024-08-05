# Streamlit + OpenAI STT(Speech-To-Text) API + GPT API를 이용한 네이버 클로바노트 클론 구현

import streamlit as st
import os
from openai import OpenAI
from pydub import AudioSegment
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

def split_wav(file_path, split_interval=480000):
    # 파일 확장자를 포함한 전체 파일명에서 확장자를 제외한 파일명만 추출
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # WAV 파일 로드
    audio = AudioSegment.from_wav(file_path)
    
    # 파일 길이(밀리초) 계산
    total_duration = len(audio)
    
    part_file_name_list = []
    # 8분 간격으로 파일 분할
    for i in range(0, total_duration, split_interval):
        # 분할할 오디오 부분 추출
        part = audio[i:i+split_interval]
        
        # 부분 파일 이름 구성
        part_file_name = f"{file_name}_part{i//split_interval + 1}.wav"
        part_file_name_list.append(part_file_name)        
        
        # 부분 파일 저장
        part.export(part_file_name, format="wav")
        print(f"Exported {part_file_name}")

    return part_file_name_list


with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[전문분야 심층인터뷰 데이터 링크](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71481)"
    uploaded_file = st.file_uploader("Upload an wav file", type=("wav"))
    stt_process = st.button(":red[텍스트 추출하기]")
    summarization_process = st.button(":blue[텍스트 요약하기]")

st.title("💬 네이버 클로바노트 클론 AI 어플리케이션")
st.caption("🚀 ClovaNote Clone by Streamlit + OpenAI STT(Speech-To-Text) API + OpenAI API")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "wav 파일을 업로드하고 텍스트 추출하기를 클릭해주세요."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if stt_process:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # OpenAI client 설정
    client = OpenAI(api_key=openai_api_key)

    # wav 파일 분할 처리 (25MB 제한을 우회하기 위해 8분을 넘어가는 wav 파일을 8분씩 나눠서 저장)
    file_name = uploaded_file.name
    with open(file_name, "wb") as file:
        file.write(uploaded_file.getvalue())
    part_file_name_list = split_wav(file_name)

    # 분할한 wav 파일들 읽어오고 OpneAI STT API로 wav 파일 안에 텍스트 추출하기
    full_text = ''
    for part_file_name in part_file_name_list:
        audio_file= open(part_file_name, "rb")
        part_transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        part_text = part_transcript.text
        full_text = full_text + part_text 
    
    st.session_state.messages.append({"role": "assistant", "content": full_text})
    st.chat_message("assistant").write(full_text)

if summarization_process:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # 긴 context를 다루기 위해 gpt-3.5-turbo-16k를 이용해서 LLM 설정
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, openai_api_key=openai_api_key)

    template = """당신은 내용을 요약하는 유용한 조수입니다. \
    내용의 주요 주제와 단락별 요약을 진행하십시오."""
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    chain = chat_prompt | llm

    summarization_result = chain.invoke({"text": st.session_state.messages[-1]['content']})
    st.chat_message("assistant").write(summarization_result.content)