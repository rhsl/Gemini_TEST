import streamlit as st
from PIL import Image ## PIL 라이브러리에서 Image 클래스를 현재 스크립트의 네임스페이스로 가져옴
import numpy as np  ## NumPy 라이브러리를 np라는 별칭으로 사용
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from google.oauth2 import service_account
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
# 환경 변수에서 JSON 파일 내용 가져오기 (Base64 인코딩된 값)
##encoded_json = "ewogICJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsCiAgInByb2plY3RfaWQiOiAiZ2VuLWxhbmctY2xpZW50LTAyMDM0NDY4MjEiLAogICJwcml2YXRlX2tleV9pZCI6ICJhNjNjYmM0MGJiNzM3MmU4NjA0YTFiMjgwZDcwZWIzZjJhNDRmMzE2IiwKICAicHJpdmF0ZV9rZXkiOiAiLS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0tXG5NSUlFdkFJQkFEQU5CZ2txaGtpRzl3MEJBUUVGQUFTQ0JLWXdnZ1NpQWdFQUFvSUJBUUMwVFpwSVl2UTlDU2tNXG5nNE5LWm5nMkxzdzRrbVZrT2Zic2hSbWhYS2M2SUdmM3VLNGpySTJYNit4Q0FHejVmZVNxTWJNdlRrV1B6MXdkXG44cGwyMzdlZVQ2Y3habUxQTTBVYWFFYVg5UngxNUlqRWZQd1NWUmtDNFJld2JWWk5CMWtEZUtBZHVEWWUxK3BrXG5oK2dxK3l3RkVpRTdPZWVxbkZVUHFDc00xdGp0UENFT1ZwZnNaUHR1MmFKZExqSEVjU1QrS3QzckR6anRGb3NUXG53bWxrSGlpcE5Zb0J6N3JyUGlacHQvZDlld2RZVDJkZDZlNEVkQTFjOEpJK2l0UlJma3V6QzlIUnpoS1VWaVJVXG5yTU1lcG5pTlBRMFpSQi81RWhNYW5WYUlyUTVOVmJ6VG5MVUFXTmFzSXZtWDh6dTlRY1ZpQ0c1bCt4NGNjVEFFXG41dEVEOVU2WEFnTUJBQUVDZ2dFQUJkYWpkeWVaSldiU3M2Q0lrNmxZeUN4Z3hEamRXL21uM2hzbDdDYzR2N1dhXG5tRUljaFhRMHk1QVhWa0IyOFpId055Q2xORDdIeFlpSjY1V3JxWUZFdHJWNVFsZHJQZ1FPS2ZUcGRaODVjalNFXG5jZW00TEl4MmpiVFFsR1VkK25XUlNOUlM5Rk5wa1RzZHBjUU8zUXdhdzA2YkpvbjcrZmJXVWFENDcrYXB5ZTRaXG5QdTRDaGY0a1ppektlck5wc0UwTWhrbXhLWUNHcjlQNEY4dEErdkNCdHZFQTdkZTNXNE5BQzltQUQ3c21ySDdkXG42b3VFdE9UWXdnU1loWm9RR3p2ZHI1QkFvTXZMam9iYmlDcWJKUE82RmNTS0szdkNZQmkybVBkdndtcDR1OGx5XG51K0tZTGpHcWJ0ZXJMZlYvM2lNT2RyalpVZHNndG9VSUFveW9rZ0RLa1FLQmdRRCtJZGd6eEhCV1ZhcVo4aWdyXG5sSUlyeDRoZVJnK0dDNzZDUXNIcDVpalpQb1pIdUJMdzNBS0thOThTN0tQb09iLzBkNDVLUGJSRitWNXoySi9xXG56Uk9PRGtpUS9RQUl0M3djdnYxTDRKTDlnZFluVDdrTmJmcEo5TU1mNWk5cWpRaE92Wmo4amppQkFjSWViU0ZOXG4zc1VnYUI5OFpiWHptMDJQTEp6R25JVXU1d0tCZ1FDMW9OalpzT1IrVEJaZGdTSFgwYUdoTEJMUUFIbWRQcGZnXG55NXl6NXpYek9HcFFSNXZmZS9MeElBK01zclpOR2p1WnU5SzdvYnFkRlhCNlZleDVza0tlMUs1U0dJZE9Vb200XG40OUZkcm9VbXRDWjN4TVJKODE0N0x1Vkl4SERWcm5lZzN6QXFUcU9XSHplOC9uZGxaZnc0V3EzUGptRmRpS3RWXG55RVBmdmNsYzBRS0JnREhYQjJUWHNNUnQxcUNNaW01Sm1PSG5Kd3ArS1FzOEFHZmhtZEE2a0daU3lka0U4OG9EXG5EQWlEc1dNdnY5R0tpZWZ6RHBmbTFCVXBHK29TWVFLV3A3QWpndjNVY0k1RkZmVTVTOG4yeUQwcG1vdHBLanpGXG5CZUk5TzR0bEJJV2NGVFlFSHgzZzhwbnljMVN1U0dyU05zenRQc2VSMXdab3ZlUkhkcFBKSEtHZkFvR0Fmc3ZqXG5sMVhlcmkwQUVCYnVRWFNmbVp6akpVS3ZwQTdtaUpDY3ZSdFVsbzl6Nk1lVkVkZStLb2R2VTVJUG9wUUZ0N1ZmXG4wSTEvY2RwZHc0bm9wS3pGeFl4RWhodUptaXdVNlhaaDJ6elN4OHBNY0tCMVNBc1daY3EvVnFXTkFCL2tjL0piXG45YTBnbHRVRVhIUnBkZWhVeENMSjVIUkpsTHFzb2g1RkJCWENWWUVDZ1lCaWhoVUlkWmtYekFyQzE1U2ppQlgyXG5CTCtCNFpNYjRwbnZ4eWRlMWJZaDJoS21vaXhaOFNKVkVMekdMOHlla014M3E5bHAybnBOMTBZU1VlZ1FFUS9pXG55dEVUeTEzaDE1S1g1Vllwc3VpQUtqUlo1S2s5UjJIa01ETUs0SEVvUFJZcnlxd1VSMFNNR0hzRTdENWhiY3BVXG4xN1YvcXN2alRDTXlLeUxMYkxpOC9nPT1cbi0tLS0tRU5EIFBSSVZBVEUgS0VZLS0tLS1cbiIsCiAgImNsaWVudF9lbWFpbCI6ICJ2ZXJ0ZXgtZ2VtaW5pLWFwaUBnZW4tbGFuZy1jbGllbnQtMDIwMzQ0NjgyMS5pYW0uZ3NlcnZpY2VhY2NvdW50LmNvbSIsCiAgImNsaWVudF9pZCI6ICIxMDE0Mjk4NTQ5MDQzMTU3MjExODAiLAogICJhdXRoX3VyaSI6ICJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20vby9vYXV0aDIvYXV0aCIsCiAgInRva2VuX3VyaSI6ICJodHRwczovL29hdXRoMi5nb29nbGVhcGlzLmNvbS90b2tlbiIsCiAgImF1dGhfcHJvdmlkZXJfeDUwOV9jZXJ0X3VybCI6ICJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9vYXV0aDIvdjEvY2VydHMiLAogICJjbGllbnRfeDUwOV9jZXJ0X3VybCI6ICJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9yb2JvdC92MS9tZXRhZGF0YS94NTA5L3ZlcnRleC1nZW1pbmktYXBpJTQwZ2VuLWxhbmctY2xpZW50LTAyMDM0NDY4MjEuaWFtLmdzZXJ2aWNlYWNjb3VudC5jb20iLAogICJ1bml2ZXJzZV9kb21haW4iOiAiZ29vZ2xlYXBpcy5jb20iCn0K"
encoded_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
st.info("encoded_json : " + encoded_json)
decoded_json = base64.b64decode(encoded_json).decode('utf-8')

# 임시 파일 생성 (필요에 따라 생략 가능)
with open('temp_credentials.json', 'w') as f:
    f.write(decoded_json)

# 임시 파일을 사용하여 Credentials 객체 생성
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'temp_credentials.json'


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

