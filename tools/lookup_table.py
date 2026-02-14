import os
import json

data_root = 'D:/Sign_Language_Dataset'
save_path = 'D:/uzh_project/IIGA/tools/data/lookup_table.json'  # Ensure this is a file path
classes = os.listdir(os.path.join(data_root, 'subject4'))

# Create a lookup table (vocabulary)
def create_lookup_table(classes, file_path):
    unique_classes = sorted(set(classes))
    lookup_table = {cls: idx for idx, cls in enumerate(unique_classes)}
    with open(file_path, 'w') as file:
        json.dump(lookup_table, file)
    return lookup_table

# Ensure the directory exists
directory = os.path.dirname(save_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# Generate the lookup table
lookup_table = create_lookup_table(classes, save_path)
print("Lookup Table:", lookup_table)
