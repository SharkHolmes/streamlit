import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
import numpy as np
import os
import streamlit as st
from create_video import video_create  # create_video.py 파일에서 video_create 함수를 가져옴
from annotation import run
from xml.etree import ElementTree as ET

def file_upload():
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the video and extract frames every 5 seconds
        frames_dir = process_video("temp_video.mp4")
        st.session_state.frames_directory = frames_dir
        return frames_dir
    return None

def process_video(video_path):
    # 출력 폴더가 존재하지 않으면 생성
    frames_dir = f"img_dir_{os.path.splitext(os.path.basename(video_path))[0]}"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # 동영상 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)

    # 첫 번째 프레임 읽기
    ret, previous_frame = cap.read()
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    frame_number = 0
    last_saved_frame = None
    saved = False

    # 이전 프레임이 존재할 때까지 반복
    while ret:
        # 현재 프레임 읽기
        ret, current_frame = cap.read()
        
        if not ret:
            break

        # 현재 프레임을 그레이스케일로 변환
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        if last_saved_frame is None:
            # 첫 번째 비교에서는 프레임을 저장하고 초기화
            frame_filename = os.path.join(frames_dir, f'frame_{frame_number}.jpg')
            cv2.imwrite(frame_filename, current_frame)
            print(f'Saved initial frame: {frame_filename}')
            last_saved_frame = current_gray.copy()
            saved = True
        else:
            # 현재 프레임과 마지막 저장된 프레임의 차이 계산
            difference = cv2.absdiff(last_saved_frame, current_gray)
            _, difference = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
            
            if np.sum(difference) == 0 and not saved:
                # 동일한 프레임이 발견되었고 이전에 저장하지 않은 경우
                frame_filename = os.path.join(frames_dir, f'frame_{frame_number}.jpg')
                cv2.imwrite(frame_filename, current_frame)
                print(f'Saved identical frame: {frame_filename}')
                saved = True
            elif np.sum(difference) != 0:
                # 다른 프레임이 나타났을 때
                saved = False
                last_saved_frame = current_gray.copy()
        
        # 프레임 번호 증가
        frame_number += 1

    # 동영상 캡처 객체 해제
    cap.release()
    return frames_dir

def update_annotation(image, filename, x_min, y_min, x_max, y_max, label):
    tree = ET.parse(filename)
    root = tree.getroot()
    for obj in root.findall('object'):
        obj.find('name').text = label
        bndbox = obj.find('bndbox')
        bndbox.find('xmin').text = str(x_min)
        bndbox.find('ymin').text = str(y_min)
        bndbox.find('xmax').text = str(x_max)
        bndbox.find('ymax').text = str(y_max)
    tree.write(filename)
    
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image

def display_images(image_paths, start_idx=0, images_per_row=20):
    end_idx = start_idx + images_per_row
    selected_images = image_paths[start_idx:end_idx]
    
    cols = st.columns(images_per_row)
    for idx, img_path in enumerate(selected_images):
        img = Image.open(img_path)
        with cols[idx]:
            st.image(img, use_column_width=True)

def load_image_paths(frames_directory):
    image_paths = sorted([os.path.join(frames_directory, f) for f in os.listdir(frames_directory) if f.endswith('.jpg')])
    return image_paths
if __name__ == "__main__":
    # 사이드바에서 페이지 선택
    page = st.sidebar.selectbox("Choose your page", ["Annotation", "위해물품 감지"])
    frames_directory = st.session_state.get('frames_directory')  # 이미 처리된 프레임 디렉토리가 있는지 확인

    if frames_directory is None:
        frames_directory = file_upload()  # 파일 업로드 및 스냅샷 생성

    st.markdown("---")

    if frames_directory:
        if "images_with_boxes" not in st.session_state:
            #video, images_with_boxes = video_create(frames_directory)
            #st.session_state.images_with_boxes = images_with_boxes
            st.session_state.current_index = 0

        if page == "Annotation":
            image_paths = load_image_paths(frames_directory)
            num_images = len(image_paths)
            images_per_row = 5

            st.write(f"총 {num_images}개의 이미지가 있습니다.")

            num_rows = (num_images + images_per_row - 1) // images_per_row
            row_slider = st.slider("Rows", 0, num_rows - 1, 0)
        
            display_images(image_paths, start_idx=row_slider * images_per_row, images_per_row=images_per_row)
            st.header("Annotation Labeling")
            custom_labels = [
    '', 'Hammer',
    'SSD',
    'Alcohol',
    'Spanner',
    'Axe',
    'Awl',
    'Throwing Knife',
    'Firecracker',
    'Thinner',
    'Plier',
    'Match',
    'Smart Phone',
    'Scissors',
    'Tablet PC',
    'Solid Fuel',
    'Bat',
    'Portable Gas',
    'Nail Clippers',
    'Knife',
    'Metal Pipe',
    'Electronic Cigarettes(Liquid)',
    'Supplementary Battery',
    'Bullet',
    'Gun Parts',
    'USB',
    'Liquid',
    'Aerosol',
    'Screwdriver',
    'Chisel',
    'Handcuffs',
    'Lighter',
    'HDD',
    'Electronic Cigarettes',
    'Battery',
    'Gun',
    'Laptop',
    'Saw',
    'Zippo Oil',
    'Stun Gun',
    'Camera',
    'Camcorder',
    'SD Card'
]

            img_path = run(frames_directory, custom_labels)

        elif page == "위해물품 감지":
            st.header("위해물품 감지")
            st.write("Use the navigation buttons to browse through the frames and update annotations.")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            if col1.button("Previous"):
                if st.session_state.current_index > 0:
                    st.session_state.current_index -= 1
            if col3.button("Next"):
                if st.session_state.current_index < len(st.session_state.images_with_boxes) - 1:
                    st.session_state.current_index += 1
            
            current_image = st.session_state.images_with_boxes[st.session_state.current_index]
            st.image(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB), use_column_width=True)

            # Add controls for annotation updates
            with st.form("annotation_form"):
                x_min = st.number_input("x_min", value=0)
                y_min = st.number_input("y_min", value=0)
                x_max = st.number_input("x_max", value=100)
                y_max = st.number_input("y_max", value=100)
                label = st.selectbox("Label", ["dog", "cat"])
                if st.form_submit_button("Update Annotation"):
                    filename = f"{frames_directory}/frame_{st.session_state.current_index * 5 * fps}.xml"
                    updated_image = update_annotation(current_image.copy(), filename, x_min, y_min, x_max, y_max, label)
                    st.session_state.images_with_boxes[st.session_state.current_index] = updated_image
                    st.image(cv2.cvtColor(updated_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
            st.markdown("---")

            # Show the video with annotations
            video = animation.FuncAnimation(
                plt.figure(figsize=(10, 6)), 
                lambda i: plt.imshow(cv2.cvtColor(st.session_state.images_with_boxes[i], cv2.COLOR_BGR2RGB)),
                frames=len(st.session_state.images_with_boxes),
                interval=1000 // 2
            )
            st.pyplot(video.to_html5_video())

            # Layout for labeled object image and label name
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Labeled Object Image")
                st.image(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB), use_column_width=True)

            with col2:
                st.subheader("Labeled Object Name")
                st.write(f"Label: {label}")
                
    else:
        st.write("Please upload a video to start.")
