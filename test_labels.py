import os

def create_dictionary(folder_path):

  dictionary = {}

  for subdir, _, files in os.walk(folder_path):
    subfolder_name = os.path.basename(subdir)
    if subfolder_name != folder_path:  # Skip the main folder
      dictionary[subfolder_name] = files

  return dictionary,dictionary.keys()

# Example usage:
folder_path = "data_moncong_sapi"
cows_dict, label_names = create_dictionary(folder_path)
label_names = list(label_names)

for name in label_names:
   globals()[name] = []

print(type(label_names))




'''
import numpy as np

# Load the NPY file into a NumPy array
data = np.load('model/output/label_dict.npy', allow_pickle=True)

# Access elements of the array
print(data)'''