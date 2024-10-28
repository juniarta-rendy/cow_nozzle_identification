import cv2
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split

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
label_id = 0

# Labels
'''
folder_path = "images/1024x1024"
label_names = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
'''
def create_dictionary(folder_path):

  dictionary = {}

  for subdir, _, files in os.walk(folder_path):
    subfolder_name = os.path.basename(subdir)
    if subfolder_name != folder_path:  # Skip the main folder
      dictionary[subfolder_name] = files

  return dictionary, list(dictionary.keys())

# Example usage:
folder_path = "data_moncong_sapi"
cows_dict, label_names = create_dictionary(folder_path)
   
n = 0

# Iterate through training images
for filename in os.listdir(train_dir):
    if filename.endswith(".jpg"):
        # Read image
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path)

        # Detect and compute features
        keypoints, des = orb.detectAndCompute(img, None)
        if des is not None:
            descriptors.append(des)
                # Mencari nomor index label
            for key, value in cows_dict.items():
                if filename in value:
                  label_index = key
                  break
            
            if label_index not in label_dict:
                label_dict[label_index] = []
                label_dict[label_index].append(n)
            else:
               label_dict[label_index].append(n)
                 
            labels.extend([n] * len(des))
            n = n + 1
print(len(descriptors))
# Convert to numpy arrays
descriptors = np.vstack(descriptors)
labels = np.array(labels)


print(len(descriptors))
