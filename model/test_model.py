import os
import random
import cv2
import numpy as np


# Load Model dan label dictionary
model_path = r"model\\output\\cowrec_knn_model.xml"
knn = cv2.ml.KNearest_load(model_path)
label_dict_path = r"model\\output\\label_dict.npy"
label_dict = np.load(label_dict_path, allow_pickle=True).item()
cows_dict_path = r"model\\output\\cows_dict.npy"
cows_dict = np.load(cows_dict_path, allow_pickle=True).item()


'''
reverse_label_dict = {v: k for k, v in label_dict.items()}
print(type(reverse_label_dict))
print(reverse_label_dict)'''

# Inisialisasi ORB
orb = cv2.ORB_create()
validation = 0
# Path ke gambar test
test_img_path = r"noseprint"

for filename in os.listdir(test_img_path):
    if filename.endswith('jpg'):
        img_val = filename
        for key,value in cows_dict.items():
            if img_val in value:
                validation_label = key
                break

# Read and preprocess the test image
        img_path = os.path.join(test_img_path,img_val)
        test_img = cv2.imread(img_path)
        test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

        # Deteksi dan compute features
        keypoints, test_des = orb.detectAndCompute(test_img_gray, None)

        # Predict menggunakan Model yang telah dilatih
        if test_des is not None:
            ret, results, neighbours, dist = knn.findNearest(test_des.astype(np.float32), k=3)
            label_counts = np.bincount(results.flatten().astype(int))
            predicted_label_id = np.argmax(label_counts)
            
            for key,value in label_dict.items():        
                if predicted_label_id == value:
                    predicted_label = key
                    break
            for key, value in cows_dict.items():
                if predicted_label in value:
                    owner = key
                    break
            if owner == validation_label:
                validation+=1

accuracy = (validation/120)*100
print(accuracy)

