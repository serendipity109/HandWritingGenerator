import os
import glob
import numpy as np
import json
import re
import itertools

wd = os.getcwd()
# common_char_source = wd + '/utils/char_source/chinese_char_std_6964.txt'
DRCD_source_folder = 'D:/training_data/keras-chinese-ocr/semantic_source/DRCD-master'
text_save_path = wd + '/utils/char_source/DRCD_sorted.txt'

file_list = glob.glob(DRCD_source_folder + '/*.json')
text = ''
for path in file_list:
    with open(path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            dic = eval(line)
            data_list = dic['data']
            for data in data_list:
                paragraphs = data['paragraphs']
                for p in paragraphs:
                    context = p['context']
                    context = re.sub(r'[A-Za-z0-9\!\%\[\]\,\。\n\t]', '', context)
                    context = re.sub(r'[^\w\s]', '', context)
                    context = context.replace(' ', '')
                    text += context

# print(len(text))  # 3765684
with open(text_save_path, 'w', encoding="utf-8") as f:
    f.write(text)
