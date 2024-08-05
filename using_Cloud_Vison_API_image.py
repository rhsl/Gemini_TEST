import io
from google.cloud import vision

# https://cloud.google.com/vision/docs/ocr?hl=ko 참조 실습
# Google Cloud Vision 클라이언트 생성 (서비스 계정 cloudvisionapi@gen-lang-client-0203446821.iam.gserviceaccount.com 사용)
client = vision.ImageAnnotatorClient.from_service_account_json('gen-lang-client-0203446821-db43aa3ad756.json')
print("a")
# 이미지 파일 열기
with io.open('receipt_sample.jpeg', 'rb') as image_file:
    content = image_file.read()
print("a")
# 이미지 객체 생성
image = vision.Image(content=content)
print("a")
# 텍스트 인식 요청
response = client.text_detection(image=image)
texts = response.text_annotations
print("a")
# 인식된 텍스트 출력
for text in texts:
    print('\n"{}"'.format(text.description))

    vertices = (['({},{})'.format(vertex.x, vertex.y)
                 for vertex in text.bounding_poly.vertices])

    print('bounds: {}'.format(','.join(vertices)))

print("a")