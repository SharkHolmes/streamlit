import os
import cv2
import pandas as pd
from tqdm import tqdm
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib import animation

def video_create(frames_directory):
    image_files = sorted([os.path.join(frames_directory, f) for f in os.listdir(frames_directory) if f.endswith('.png')])
    annotation_files = sorted([os.path.join(frames_directory, f) for f in os.listdir(frames_directory) if f.endswith('.xml')])
    
    # Check if annotation files exist for all images
    if len(image_files) != len(annotation_files):
        print("Not all images have corresponding annotation files. Skipping video creation.")
        return None, None
    
    BOX = pd.DataFrame()

    for ann_file in annotation_files:
        try:
            tree = ET.parse(ann_file)
            root = tree.getroot()
            filename = root.find('filename').text
            for obj in root.findall('object'):
                label = obj.find('name').text
                bndbox = obj.find('bndbox')
                x_min = int(bndbox.find('xmin').text)
                y_min = int(bndbox.find('ymin').text)
                x_max = int(bndbox.find('xmax').text)
                y_max = int(bndbox.find('ymax').text)
                BOX = BOX.append({
                    'filename': filename,
                    'label': label,
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max
                }, ignore_index=True)
        except Exception as e:
            print(f"Failed to load {ann_file}: {e}")

    def draw_box(image, filename):
        box = BOX[BOX['filename'] == filename]
        
        for _, row in box.iterrows():
            x_min = row['x_min']
            y_min = row['y_min']
            x_max = row['x_max']
            y_max = row['y_max']
            
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return image

    images_with_boxes = [draw_box(cv2.imread(img_path), os.path.basename(img_path)) for img_path in tqdm(image_files)]

    def create_animation(ims):
        fig = plt.figure(figsize=(10, 6))
        plt.axis('off')
        
        im = plt.imshow(cv2.cvtColor(ims[0], cv2.COLOR_BGR2RGB))
        plt.close()
        
        def animate_func(i):
            im.set_data(cv2.cvtColor(ims[i], cv2.COLOR_BGR2RGB))
            return [im]

        return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000//2)

    video = create_animation(images_with_boxes)
    return video, images_with_boxes


# Example usage:
# video, images_with_boxes = video_create("path/to/frames_directory")
