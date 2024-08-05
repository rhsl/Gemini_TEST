import streamlit as st
import google.generativeai as genai
from PIL import Image ## PIL 라이브러리에서 Image 클래스를 현재 스크립트의 네임스페이스로 가져옴
import numpy as np  ## NumPy 라이브러리를 np라는 별칭으로 사용
## google.generativeai 라이브러리를 활용한 샘플
safety_settings=[
        # {
        #     "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        #     "threshold": "BLOCK_NONE",
        # },
        # {
        #     "category": "HARM_CATEGORY_HATE_SPEECH",
        #     "threshold": "BLOCK_NONE",
        # },
        # {
        #     "category": "HARM_CATEGORY_HARASSMENT",
        #     "threshold": "BLOCK_NONE",
        # },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        }
    ]


with st.sidebar:
    # session_state 초기화
    if "image" not in st.session_state: ## st.session_state => 세션 상태를 저장하는 딕셔너리. 닥쇼노라애 "image" 라는 키 값이 존재하지 않으면
        st.session_state.image = None  ## "image" 라는 키를 생성하고 , 그 값을 None 으로 설정
    
    "[샘플 이미지 링크](https://t0.gstatic.com/licensed-image?q=tbn:ANd9GcQ_Kevbk21QBRy-PgB4kQpS79brbmmEG7m3VOTShAn4PecDU5H5UxrJxE3Dw1JiaG17V88QIol19-3TM2wCHw)"
    gemini_api_key = st.text_input("Gemini API Key", key="chatbot_api_key", type="password")
    # 이미지 파일 업로드 하여 바이트 데이터를 저장
    img_file_buffer = st.file_uploader("이미지 파일을 업로드", type=["png", "jpg", "jpeg"])

    # 이미지 파일 읽어오기
    if img_file_buffer != None:
        image = Image.open(img_file_buffer) ## Pillow 이미지 객체로 저장 (픽셀정보, 크키, 형식등에 대한 다양한 속성과 메소드가 제공됨)
        img_array = np.array(image) ## pillow 이미지 객체를 NumPy 배열로 변환. 이미지 각 픽셀값을 행렬 형태로 저장
    
        st.session_state.image = image

st.title("💬 Google Gemini Chatbot (Image 상호작용 기능 추가)")
st.caption("🚀 A streamlit chatbot powered by Google Gemini LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이미지 화면에 그리기
if st.session_state.image != None:
    st.image( ## 이미지 출력 함수
        image,
        caption=f"You amazing image has shape {img_array.shape[0:2]}", ## Image 의 높이 , 너비 정보 반환. Image 객체를 배열로 변환하면 높이, 너비, 채널 의 배열로 저장함
        use_column_width=True, ## 이미지를 화면 전체 너비에 맞추어 표시해라
    )

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["parts"][-1])

if prompt := st.chat_input():
    if not gemini_api_key:
        st.info("계속 진행하기 위해서는 당신의 Google Gemini API 키를 입력해주세요.")
        st.stop()

    genai.configure(api_key=gemini_api_key)
    # gemini-pro-vision 모델 설정
    model = genai.GenerativeModel('gemini-1.5-flash', safety_settings)

    st.session_state.messages.append({"role": "user", "parts": [prompt]})
    st.chat_message("user").write(prompt)

    # prompt와 함께 이미지를 같이 모델에 전달
    response = model.generate_content([prompt, image])
    print (response)
    msg = response.text
 
    
    st.session_state.messages.append({"role": "model", "parts": [msg]})
    st.chat_message("model").write(msg)