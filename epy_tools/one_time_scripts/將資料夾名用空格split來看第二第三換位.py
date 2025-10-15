import os
import shutil

# Set the path to your directory
directory = 'projects/SMAC/1006_state_DsrAndFas_FORMAL_sacred'

# If non-existent, exit
if not os.path.exists(directory):
    print(f"Directory '{directory}' does not exist.")
    exit(1)

# List directory content
folder_names = os.listdir(directory)

# Create the new folder names with second and third elements swapped
new_folder_names = []
for folder in folder_names:
    parts = folder.split()
    if len(parts) >= 3:
        # Swap the second and third parts
        parts[1], parts[2] = parts[2], parts[1]
        new_folder_names.append(" ".join(parts))

# Print or rename folders here
for old, new in zip(folder_names, new_folder_names):
    print(f'Renaming: {old} -> {new}')
    # Use shutil.move() to rename
    shutil.move(os.path.join(directory, old), os.path.join(directory, new))

