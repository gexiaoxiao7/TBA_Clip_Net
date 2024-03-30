import os
import csv
import random
import shutil
import rarfile
from collections import defaultdict
import random

PATH = 'E:/data'


def split_dataSet(dataSet, split_ratio=0.3):

    data_set_path = PATH + '/' + dataSet + '/' + dataSet + '_dataset.txt'
    with open(data_set_path, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)

    train_size = int(len(lines) * split_ratio)
    train_lines = lines[:train_size]
    val_lines = lines[train_size:]

    train_path = PATH + '/' + dataSet + '/' + 'train.txt'
    with open(train_path, 'w') as f:
        f.writelines(train_lines)
    print('train.txt has been created successfully!')

    val_path = PATH + '/' + dataSet + '/' + 'test.txt'
    with open(val_path, 'w') as f:
        f.writelines(val_lines)
    print('val.txt has been created successfully!')


def extract_shots(dataSet, shots, dataset_path, shots_path):
    label_path = PATH + '/' + dataSet + '/' + dataSet + '_labels.csv'
    folder_id_map = {}
    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip column names
        for row in reader:
            folder_id_map[row[1]] = row[0]

    with open(dataset_path, 'r') as f:
        lines = f.readlines()

    files_by_label = defaultdict(list)
    for line in lines:
        file, label = line.strip().split()
        files_by_label[label].append(file)

    train_path = 'train_' + shots_path
    with open(train_path, 'w') as f:
        for label, files in files_by_label.items():
            selected_files = random.sample(files, min(shots, len(files)))
            for file in selected_files:
                f.write(f'{file} {label}\n')

    val_path = 'val_' + shots_path
    with open(val_path, 'w') as f:
        for label, files in files_by_label.items():
            # Select half the number of training samples for validation
            selected_files = random.sample(files, min(shots // 4, len(files)))
            for file in selected_files:
                f.write(f'{file} {label}\n')

    print(f'{shots}shots.txt has been created successfully!')
def split_file(file_path,num):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Calculate the size of each chunk
    chunk_size = len(lines) // num

    # Split the data into four chunks
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    # Write each chunk to a new file
    for i, chunk in enumerate(chunks, 1):
        with open(f'test_split_{i}.txt', 'w') as f:
            f.writelines(chunk)

    print('DONE')
# Call the function with the path to your file


if __name__ == '__main__':
    # split_dataSet('hmdb51_org')
    # split_dataSet('UCF-101')
    extract_shots('UCF-101', 4, 'D:\gfx\CODE\project\TBA_Clip_Net\datasets_splits\\UCF-101\\train.txt',
                  'shot2.txt')

    pass
