import os

# Base data folder path (adjust if needed)
base_path = os.path.join(os.getcwd(), 'data')

# Total 36 folders: 0-9 for digits, 10-35 for A-Z
for i in range(36):
    folder_path = os.path.join(base_path, str(i))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")
