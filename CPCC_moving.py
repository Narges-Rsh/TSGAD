import os
import shutil

# Directory containing the .npy files
source_directory = '/home/galinezh/TCVAE_modified/data/CPCC/gt/test_frame_mask'

dest = '/home/galinezh/TCVAE_modified/data/CPCC'


# List all .npy files in the source directory
files = [file for file in os.listdir(source_directory) if file.endswith('.npy')]

# Iterate over each .npy file
for filename in files:
    # Extract the value of 'a' from the filename
    camera = int(filename.split('_')[0])
    dest_path = dest + str(camera)
    dest_path = os.path.join(dest + str(camera), 'gt/test_frame_mask/')

    # Copy the file to the destination directory
    shutil.copyfile(os.path.join(source_directory, filename), os.path.join(dest_path, filename))


source_directory = '/home/galinezh/TCVAE_modified/data/CPCC/pose/test'

dest = '/home/galinezh/TCVAE_modified/data/CPCC'


# List all .npy files in the source directory
files = [file for file in os.listdir(source_directory) if file.endswith('.json')]

# Iterate over each .npy file
for filename in files:
    # Extract the value of 'a' from the filename
    camera = int(filename.split('_')[0])
    dest_path = dest + str(camera)
    dest_path = os.path.join(dest + str(camera), 'pose/test/')

    # Copy the file to the destination directory
    shutil.copyfile(os.path.join(source_directory, filename), os.path.join(dest_path, filename))


source_directory = '/home/galinezh/TCVAE_modified/data/CPCC/pose/train'

dest = '/home/galinezh/TCVAE_modified/data/CPCC'


# List all .npy files in the source directory
files = [file for file in os.listdir(source_directory) if file.endswith('.json')]

# Iterate over each .npy file
for filename in files:
    # Extract the value of 'a' from the filename
    camera = int(filename.split('_')[0])
    dest_path = dest + str(camera)
    dest_path = os.path.join(dest + str(camera), 'pose/train/')

    # Copy the file to the destination directory
    shutil.copyfile(os.path.join(source_directory, filename), os.path.join(dest_path, filename))