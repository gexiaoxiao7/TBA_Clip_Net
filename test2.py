# Step 1: Read the file
with open('datasets_splits/TBAD-8/test.txt', 'r') as f:
    lines = f.readlines()

# Step 2: Classify the data by label
data = {str(i): [] for i in range(8)}
for line in lines:
    label = line.strip().split()[-1]
    data[label].append(line)

# Step 3: Reorder the data
new_data = []
while any(data.values()):
    for label in data:
        if data[label]:
            new_data.append(data[label].pop(0))

# Step 4: Write the new data to a new file
with open('datasets_splits/TBAD-8/test_reordered.txt', 'w') as f:
    f.writelines(new_data)


# Step 1: Read the file
with open('datasets_splits/TBAD-8/test_reordered.txt', 'r') as f:
    lines = f.readlines()

# Step 2 and 3: Split the data and write to new files
for i in range(0, len(lines), 60):
    part = lines[i:i+60]
    with open(f'datasets_splits/TBAD-8/test_reordered_part{i//60+1}.txt', 'w') as f:
        f.writelines(part)