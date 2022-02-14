import os
from opencc import OpenCC

wd = os.getcwd()
name_source_path = 'D:/training_data/keras-chinese-ocr/semantic_source/Chinese_Names.txt'
text_save_path = wd + '/utils/char_source/chinese_name.txt'

cc = OpenCC('s2twp')

text = ''
with open(name_source_path, 'r', encoding="utf-8") as f:
    data_list = f.readlines()
    for data in data_list:
        text += data

text = cc.convert(text)
text = text.replace('\n', '')
with open(text_save_path, 'w', encoding="utf-8") as f:
    f.write(text)

print(len(text))  # 4407853
text = list(text)
text = list(set(text))
print(len(text))  # 2347


