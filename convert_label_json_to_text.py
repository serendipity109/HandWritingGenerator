import os
import json

wd = os.getcwd()
label_path = '/Users/dennislin/Downloads/labels.json'

with open(label_path, 'r') as f:
    data = json.load(f)

labels = data['labels']

lines = []
for key in labels:
    index = key
    text = labels[key]
    lines.append(index + " " + text + '\n')

with open (wd + '/text.txt', 'w') as fp:
    fp.writelines(lines)

