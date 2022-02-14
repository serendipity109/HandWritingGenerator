import os
import numpy as np

wd = os.getcwd()
road_name_source_path = 'D:/training_data/keras-chinese-ocr/semantic_source/All_Road_Data.json'

numbers_to_char_dict = {'0': '十', '1': '一', '2': '二', '3': '三', '4': '四',
                        '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}

numbers_replace_dict = {'１': '1', '２': '2', '３': '3', '４': '4', '５': '5',
                        '６': '6', '７': '7', '８': '8', '９': '9', '０': '0'}

text_save_path = wd + '/utils/char_source/gen_road_name.txt'

text = ''
with open(road_name_source_path, 'r', encoding="utf-8") as f:
    all_data = f.readlines()
    all_data = eval(all_data[0])
    for data in all_data:
        np.random.seed()
        city_name = data['CityName']
        area_list = data['AreaList']
        for area in area_list:
            area_name = area['AreaName']
            road_list = area['RoadList']
            for road in road_list:
                # 號
                n1 = str(np.random.randint(1, 500))
                # 樓
                n2 = str(np.random.randint(0, 10))
                # 之
                n3 = str(np.random.randint(1, 10))
                temp_text = ''
                road_name = road['RoadName']
                for c in numbers_replace_dict.keys():
                    road_name = road_name.replace(c, numbers_replace_dict[c])
                temp_text += city_name + area_name + road_name + n1 + '號' + numbers_to_char_dict[n2] + '樓之' + n3
                text += temp_text
# print(len(text))  # 750591
with open(text_save_path, 'w', encoding="utf-8") as f:
    f.write(text)



