import streamlit as st
from PIL import Image ## PIL 라이브러리에서 Image 클래스를 현재 스크립트의 네임스페이스로 가져옴
import numpy as np  ## NumPy 라이브러리를 np라는 별칭으로 사용
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from google.oauth2 import service_account
from PyPDF2 import PdfReader ##pip install PyPDF2
import os

# https://console.cloud.google.com/vertex-ai/generative/multimodal/prompt-examples/Image%20text%20to%20JSON?hl=ko&project=gen-lang-client-0203446821
# 참조 실습
# vertex-gemini-api@gen-lang-client-0203446821.iam.gserviceaccount.com 서비스 계정의 json 을 사용함 (VertexAI 관리자 역할 부여 해야 함)


def generate(userMsg):
  
  responses = model.generate_content(
      #[image1, """이미지의 텍스트 데이터를 key : value 형태로 json 으로 추출 해줘"""],
      #[image1, """총 수납금액이 얼마야?"""],
      [imageModel, userMsg],
      generation_config=generation_config,
      safety_settings=safety_settings,
      stream=True,
  )
  return responses
#   for response in responses:
#       print(response.text, end="")
    
def check_file_mymetype(uploaded_file):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension.lower() in ["pdf"]:
            return "application/pdf"
        elif file_extension.lower() in ["jpg", "jpeg", "png", "gif"]:
            return "image/"+file_extension.lower()
        else:
            return "Unknown"
    else:
        return "No file uploaded"


# 서비스 계정 JSON 파일 경로
##service_account_json = "gen-lang-client-0203446821-a63cbc40bb73.json"
## 로컬에서 실행 할때
##os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/gonihyun/TEST/Gemini/Gemini_TEST/gen-lang-client-0203446821-a63cbc40bb73.json"
## streamlit 에서 실행 할때
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

# 서비스 계정 정보 로드
# credentials = service_account.Credentials.from_service_account_file(
#service_account_json,
#scopes=["https://www.googleapis.com/auth/cloud-platform"])

vertexai.init(project="gen-lang-client-0203446821", location="us-central1")
model = GenerativeModel(
"gemini-1.5-flash-001",
)

# image_path = "receipt_sample.jpeg"
# # 이미지를 base64로 인코딩
# with io.open(image_path, "rb") as image_file:
#     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# image1 = Part.from_data(
#     mime_type="image/jpeg",
#     data=base64.b64decode(encoded_string)
# )

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}



with st.sidebar:
    # session_state 초기화
    if "image" not in st.session_state: ## st.session_state => 세션 상태를 저장하는 딕셔너리. 닥쇼노라애 "image" 라는 키 값이 존재하지 않으면
        st.session_state.image = None  ## "image" 라는 키를 생성하고 , 그 값을 None 으로 설정

    gemini_api_key = st.text_input("Gemini API Key", key="chatbot_api_key", type="password")
    # 이미지 파일 업로드 하여 바이트 데이터를 저장
    img_file_buffer = st.file_uploader("이미지 파일을 업로드", type=["png", "jpg", "jpeg","pdf"])

# 이미지 파일 읽어오기
    if img_file_buffer != None:
        file_mymetype = check_file_mymetype(img_file_buffer)

        encoded_string = base64.b64encode(img_file_buffer.read()).decode('utf-8')
        imageModel = Part.from_data(
            mime_type=file_mymetype,
            data=base64.b64decode(encoded_string)
        )
        if file_mymetype.startswith("image") :
            image = Image.open(img_file_buffer) ## Pillow 이미지 객체로 저장 (픽셀정보, 크키, 형식등에 대한 다양한 속성과 메소드가 제공됨)
            img_array = np.array(image) ## pillow 이미지 객체를 NumPy 배열로 변환. 이미지 각 픽셀값을 행렬 형태로 저장
    
            st.session_state.image = image

st.title("💬 Google VertexAI Chatbot")
st.caption("🚀 이미지에서 데이터 추출하기")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "이미지를 첨부하고 원하는 데이터 결과를 입력해주세요"}]

# 이미지 화면에 그리기
if st.session_state.image != None:
    st.image( ## 이미지 출력 함수
        image,
        caption=f"You amazing image has shape {img_array.shape[0:2]}", ## Image 의 높이 , 너비 정보 반환. Image 객체를 배열로 변환하면 높이, 너비, 채널 의 배열로 저장함
        use_column_width=True, ## 이미지를 화면 전체 너비에 맞추어 표시해라
    )


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not gemini_api_key:
        st.info("계속 진행하기 위해서는 당신의 Google Gemini API 키를 입력해주세요.")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})  ## 입력 받은 내용을 추가
    st.chat_message("user").write(prompt) # 입력 받은 내용을 화면에 표시
    userMsg = st.session_state.messages[-1]['content'] #st.session_state 에 저장한 메시지를 변수에 저장
   ## responses = model.generate_content(userMsg)  ## AI 모델에게 메시지 전달 
    responses = generate(userMsg)

    msg =""
    for response in responses:
        msg += response.text
    
    st.session_state.messages.append({"role": "assistant", "content": msg}) ## 응답을 st.session_state 에 저장
    st.chat_message("assistant").write(msg) ## assistant 의 응답으로 화면에 표시

