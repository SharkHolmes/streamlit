import streamlit as st
import cv2
from PIL import Image
import numpy as np

# 제목 설정
st.set_page_config(page_title="X-ray Video Object Detection")
st.title("X-ray Video Object Detection")

# 동영상 업로드
uploaded_file = st.file_uploader("Upload an X-ray video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # 동영상 읽기
    video = cv2.VideoCapture(uploaded_file.name)

    # 프레임 추출 및 객체 인식
    screenshots = []
    labels = []
    for i in range(5):
        ret, frame = video.read()
        if ret:
            # 객체 인식 모델 적용 (예시로 OpenCV 사용)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 스크린샷 캡처 및 바운딩 박스 그리기
            screenshot = frame.copy()
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(screenshot, (x, y), (x+w, y+h), (0, 255, 0), 2)

            screenshots.append(screenshot)
            labels.append([])

    # 결과 이미지 및 레이블 표시
    st.subheader("Detected Objects")
    cols = st.columns(5)
    for i, screenshot in enumerate(screenshots):
        with cols[i]:
            st.image(screenshot, caption=f"Screenshot {i+1}", use_column_width=True)
            label = st.text_input(f"Label for Screenshot {i+1}", key=f"label_{i+1}")
            labels[i].append(label)

    video.release()
    cv2.destroyAllWindows()
