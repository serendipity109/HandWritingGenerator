import os
import glob
import json


wd = os.getcwd()
char_path = wd + '/utils/chinese_char_std_6964.txt'
data_folder = '/home/sightour/Desktop/training_data/handwrite_utils/THCD_data_all/cleaned_data'
img_list_save_path = wd + '/utils/std_6964_image_list.json'


def create_char_image_list():
    char_img_dict = {}
    with open(char_path, 'r', encoding='utf-8') as f:
        char_list = f.read()
    char_list = ''.join(char_list).replace('\n', '')
    print(len(char_list))
    for c in char_list:
        img_list = glob.glob(os.path.join(data_folder, '*', c + '*.png'))
        if len(img_list) != 0:
            char_img_dict[c] = img_list
        else:
            print(c)
    with open(img_list_save_path, 'w', encoding='utf-8') as outfile:
        json.dump(char_img_dict, outfile, indent=4, ensure_ascii=False)


create_char_image_list()
