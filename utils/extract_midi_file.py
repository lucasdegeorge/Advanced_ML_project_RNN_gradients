#%% 
import os
import shutil
from sklearn.model_selection import train_test_split

def organize_files(source_folder, destination_folder, train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15):
    # Create train, eval, and test folders
    train_folder = os.path.join(destination_folder, 'train')
    eval_folder = os.path.join(destination_folder, 'eval')
    test_folder = os.path.join(destination_folder, 'test')

    for folder in [train_folder, eval_folder, test_folder]:
        os.makedirs(folder, exist_ok=True)

    # List all files in the source folder
    all_files = [os.path.join(root, file) for root, _, files in os.walk(source_folder) for file in files if files != [] ]
    print(len(all_files))

    # Use train_test_split to split files into train, eval, and test sets
    train_files, test_and_eval_files = train_test_split(all_files, test_size=(eval_ratio + test_ratio), random_state=42)
    eval_files, test_files = train_test_split(test_and_eval_files, test_size=test_ratio / (eval_ratio + test_ratio), random_state=42)
    print("split ok")

    # Move files to the respective folders
    for file in train_files:
        destination_path = os.path.join(train_folder, file.split('\\')[-1])
        shutil.copy(file, destination_path)

    print("train ok")

    for file in eval_files:
        destination_path = os.path.join(eval_folder, file.split('\\')[-1])
        shutil.copy(file, destination_path)

    print("eval ok")

    for file in test_files:
        destination_path = os.path.join(test_folder, file.split('\\')[-1])
        shutil.copy(file, destination_path)

# Example usage:
source_directory = "C:/Users/lucas/Documents/GitHub/RNN_gradient_problem/raw_data/adl-piano-midi"
destination_directory = "C:/Users/lucas/Documents/GitHub/RNN_gradient_problem/raw_data/midi"

organize_files(source_directory, destination_directory)