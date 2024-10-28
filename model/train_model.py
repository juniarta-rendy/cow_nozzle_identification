import cv2
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2

# Directory containing training images
train_dir = r"noseprint"
model_dir = r"model/output"

# Ensure the model directory exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Initialize ORB detector
orb = cv2.ORB_create()

# Initialize lists for descriptors and labels
descriptors = []
labels = []
label_dict = {}
recap_label = {} 
label_id = 0

# Labels
folder_path = r'images/1024x1024'
images_name = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

def create_dictionary(folder_path):

  dictionary = {}

  for subdir, _, files in os.walk(folder_path):
    subfolder_name = os.path.basename(subdir)
    if subfolder_name != folder_path:  # Skip the main folder
      dictionary[subfolder_name] = files

  return dictionary, list(dictionary.keys())

# dictionary nama pemilik dan sapi
folder_path = "data_moncong_sapi"
cows_dict, label_names = create_dictionary(folder_path)

# menyimpan dictionary daftar sapi
cows_dict_path = os.path.join(model_dir, "cows_dict.npy")
np.save(cows_dict_path,cows_dict)
   
n = 0

# loop untuk tiap file gambar
for filename in os.listdir(train_dir):
    if filename.endswith(".jpg"):
        # membaca gambar
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path)

        # deteksi dan mengidentifikasi 'feature'
        keypoints, des = orb.detectAndCompute(img, None)
        if des is not None:
            descriptors.append(des)
            # Mencari nomor index label
            for key, value in cows_dict.items():
                if filename in value:
                  label_index = key
                  break
            
            if label_index not in recap_label:
                recap_label[label_index] = []
                recap_label[label_index].append(n)

            if n not in label_dict:
               label_dict[images_name[n]] = n
            labels.extend([n] * len(des))
            n = n + 1 

# nomor ID nama pemilik
id_dict = {x: id_value for x, id_value in enumerate(recap_label.values())}

# Convert to numpy arrays
descriptors = np.vstack(descriptors)
labels = np.array(labels)
new_labels = []

# ubah id pada "labels" menjadi nama pemilik sapi
i = 0
for a in labels:
  for key, value in id_dict.items():
     if a in value:
        new_labels.append(key)
  n+=1


# Train k-NN classifier
knn = cv2.ml.KNearest_create()
knn.train(descriptors.astype(np.float32), cv2.ml.ROW_SAMPLE, labels)

# Save model yang telah dilatih dan label dictionary
model_path = os.path.join(model_dir, "cowrec_knn_model.xml")
knn.save(model_path)
label_dict_path = os.path.join(model_dir, "label_dict.npy")
np.save(label_dict_path, label_dict)


print(f"Train Model selesaidan Model disimpan.")