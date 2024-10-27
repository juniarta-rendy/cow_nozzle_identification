import os

def create_dictionary(folder_path):

  dictionary = {}

  for subdir, _, files in os.walk(folder_path):
    subfolder_name = os.path.basename(subdir)
    if subfolder_name != folder_path:  # Skip the main folder
      dictionary[subfolder_name] = files

  return dictionary

# Example usage:
folder_path = "data_moncong_sapi"
cow_images_dict = create_dictionary(folder_path)

print(cow_images_dict.keys())
print(cow_images_dict)




'''
import numpy as np
s
# Load the NPY file into a NumPy array
data = np.load('model/output/label_dict.npy', allow_pickle=True)

# Access elements of the array
print(data)'''