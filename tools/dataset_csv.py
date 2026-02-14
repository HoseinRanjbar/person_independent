import os
import csv
import random
import pandas as pd

def create_and_split_dataset_csv(dataset_dir, train_csv, test_csv, val_csv, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    # Initialize an empty list to store dataset entries
    data_entries = []

    # Unique identifier for each video
    signers = ['subject{}'.format(idx) for idx in range(1,11)]

    print(signers)
    # Traverse through each signer folder
    for signer in signers:
        signer_path = os.path.join(dataset_dir, signer)

        # Ensure we are processing directories only
        if os.path.isdir(signer_path):
            # Traverse through each class folder within the signer folder
            for class_name in os.listdir(signer_path):
                class_path = os.path.join(signer_path, class_name)

                if os.path.isdir(class_path):
                    # Traverse through each video file in the class folder
                    for video_file in os.listdir(class_path):
                        # Construct the full path of the video file
                        video_path = os.path.join(class_path, video_file)

                        # Check if it's a file
                        if os.path.isfile(video_path):
                            # Create a new entry
                            video_id = os.path.join(signer,class_name,video_file[:-4])
                            data_entries.append({
                                'id': video_id,
                                'video_path': os.path.join(signer,class_name,video_file),
                                'signer': signer,
                                'class': class_name
                            })

    # Shuffle the data
    random.shuffle(data_entries)

    # Calculate the number of samples for each split
    total_samples = len(data_entries)
    train_size = int(train_ratio * total_samples)
    test_size = int(test_ratio * total_samples)
    val_size = total_samples - train_size - test_size

    # Split the dataset
    train_entries = data_entries[:train_size]
    test_entries = data_entries[train_size:train_size + test_size]
    val_entries = data_entries[train_size + test_size:]

    # Helper function to save entries to a CSV file
    def save_to_csv(entries, csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['id', 'video_path', 'signer', 'class']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                writer.writerow(entry)

    # Save the splits to separate CSV files
    save_to_csv(train_entries, train_csv)
    save_to_csv(test_entries, test_csv)
    save_to_csv(val_entries, val_csv)

    print(f"Train CSV file created successfully at: {train_csv}")
    print(f"Test CSV file created successfully at: {test_csv}")
    print(f"Validation CSV file created successfully at: {val_csv}")

# Usage example:
dataset_dir = "D:/Sign_Language_Dataset"

train_ratio = 0.7  # 70% for training
test_ratio = 0.2   # 20% for testing
val_ratio = 0.1    # 10% for validation

train_csv = "D:/Sign_Language_Dataset/train_dataset.csv"
test_csv = "D:/Sign_Language_Dataset/test_dataset.csv"
val_csv = "D:/Sign_Language_Dataset/val_dataset.csv"

# Create and split the dataset into train, test, and validation CSV files
create_and_split_dataset_csv(dataset_dir, train_csv, test_csv, val_csv, train_ratio, test_ratio, val_ratio)
