import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image
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
        frames_dir = extract_frames("temp_video.mp4", interval=5)
        st.session_state.frames_directory = frames_dir
        return frames_dir
    return None

def extract_frames(video_path, interval):
    # Load the video using OpenCV
    video = cv2.VideoCapture(video_path)
    
    # Get the frames per second (fps) of the video
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    # Create a directory to save frames if it doesn't exist
    frames_dir = f"img_dir_{os.path.splitext(os.path.basename(video_path))[0]}"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    # Calculate the interval in terms of frames
    frame_interval = interval * fps
    frame_count = 0
    success, frame = video.read()
    while success:
        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_number % frame_interval == 0:
            img_path = f"{frames_dir}/frame_{frame_number}.png"
            cv2.imwrite(img_path, frame)
            frame_count += 1
            st.write(f"Saved frame at {frame_number // fps} seconds as {img_path}")
        success, frame = video.read()
    
    st.success(f"Extracted {frame_count} frames saved to '{frames_dir}/' directory.")
    video.release()
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
    image_paths = sorted([os.path.join(frames_directory, f) for f in os.listdir(frames_directory) if f.endswith('.png')])
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
            video, images_with_boxes = video_create(frames_directory)
            st.session_state.images_with_boxes = images_with_boxes
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
            custom_labels = ["", "dog", "cat"]
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
