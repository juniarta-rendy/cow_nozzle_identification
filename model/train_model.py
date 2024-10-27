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
'''folder_path = "data_moncong_sapi"
label_names = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]'''
def create_dictionary(folder_path):

  dictionary = {}

  for subdir, _, files in os.walk(folder_path):
    subfolder_name = os.path.basename(subdir)
    if subfolder_name != folder_path:  # Skip the main folder
      dictionary[subfolder_name] = files

  return dictionary

# Example usage:
folder_path = "data_moncong_sapi"
label_names = create_dictionary(folder_path)

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
            if n not in label_dict:
                label_dict[label_names[n]] = n
            labels.extend([n] * len(des))
            n = n + 1

# Convert to numpy arrays
descriptors = np.vstack(descriptors)
labels = np.array(labels)

# Split the data into training and test sets (80% train, 20% test)
des_train, des_test, labels_train, labels_test = train_test_split(descriptors, labels, test_size=0.2, random_state=42)

# Train k-NN classifier
knn = cv2.ml.KNearest_create()
knn.train(descriptors.astype(np.float32), cv2.ml.ROW_SAMPLE, labels)

# Save the trained model and label dictionary
model_path = os.path.join(model_dir, "cowrec_knn_model.xml")
knn.save(model_path)
label_dict_path = os.path.join(model_dir, "label_dict.npy")
np.save(label_dict_path, label_dict)

# Evaluate accuracy on the test set
ret, results, neighbours, dist = knn.findNearest(des_test.astype(np.float32), k=3)
correct_predictions = np.sum(results.flatten() == labels_test)
accuracy = correct_predictions / len(labels_test) * 100

print(f"Training completed and model saved. Test Accuracy: {accuracy:.2f}%")