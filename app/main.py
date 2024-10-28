import os
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import numpy as np
import time

# Load train model dan label dictionary
model_path = r'model//output//cowrec_knn_model.xml'
knn = cv2.ml.KNearest_load(model_path)
label_dict_path = r'model//output//label_dict.npy'
label_dict = np.load(label_dict_path, allow_pickle=True).item()
reverse_label_dict = {v: k for k, v in label_dict.items()}
owner_dict_path = r'model//output//cows_dict.npy'
owner_dict = np.load(owner_dict_path, allow_pickle=True).item()

# ORB detector
orb = cv2.ORB_create()

def convert_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    adaptive_thresh = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    inverted_image = cv2.bitwise_not(adaptive_thresh)
    
    # Apply morphological operations to enhance the patterns (optional)
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)

    # Resize the processed image to 256x256 (as used for model input)
    processed_image = cv2.resize(processed_image, (256, 256))

    return processed_image

# Function to handle file drop
def on_drop(event):
    file_path = event.data.strip('{}')  # Handle dropped file path

    # Process the dropped image
    original_image, processed_image = process_image(file_path)

    # Display images
    display_image(original_image, original_label)
    display_image(processed_image, processed_label)

    # Predict the class of the processed image using ORB and KNN
    keypoints, test_des = orb.detectAndCompute(processed_image, None)

    if test_des is not None:
        ret, results, neighbours, dist = knn.findNearest(test_des.astype(np.float32), k=3)
        label_counts = np.bincount(results.flatten().astype(int))
        predicted_label_id = np.argmax(label_counts)

        predicted_label = reverse_label_dict[predicted_label_id]
        for key,value in owner_dict.items():
            if predicted_label in value:
                predicted_label = key
                break
        result_text = f"Sapi Milik: {predicted_label}"
        result_label.config(text=result_text, bg='green', font=("Helvetica", 20))
    else:
        result_label.config(text="Tidak terdaftar", bg='red', font=("Helvetica", 20))

# Function to process the image (without nose detection)
def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Tidak dapat memuat gambar: {image_path}")

    # Directly process the entire image without nose detection
    processed_image = convert_image(image)

    return image, processed_image

# Function to display an image on a label
def display_image(cv_img, label):
    # Convert the image to RGB format
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_img = Image.fromarray(cv_img_rgb)
    # Resize the image to fit the label
    pil_img = pil_img.resize((250, 250), Image.LANCZOS)
    # Convert to ImageTk
    imgtk = ImageTk.PhotoImage(image=pil_img)
    label.config(image=imgtk)
    label.image = imgtk

# Set up the GUI
root = TkinterDnD.Tk()
root.title("Rendy Tampan")
root.geometry("800x600")

# Labels to display images
original_label = tk.Label(root)
original_label.grid(row=0, column=0, padx=10, pady=10)
processed_label = tk.Label(root)
processed_label.grid(row=0, column=2, padx=10, pady=10)

# Label to display the result
result_label = tk.Label(root, text="Hasil akan terlihat di sini", font=("Helvetica", 16), width=40, height=2)
result_label.grid(row=2, column=0, columnspan=3, pady=10)

# Instructions
instructions = tk.Label(root, text="Tarik dan letakkan file gambar.", font=("Helvetica", 16))
instructions.grid(row=1, column=0, columnspan=3, pady=10)

# Enable drag and drop
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_drop)

# Start the GUI event loop
root.mainloop()
