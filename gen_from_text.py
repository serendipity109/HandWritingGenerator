import traceback
import os
import glob
import time
from itertools import repeat
import math_utils
import effects
import numpy as np
import multiprocessing as mp
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageFont, ImageDraw
from random import shuffle
import effect_utils
from fontTools.ttLib import TTFont


wd = os.getcwd()
semantic_source = wd + '/utils/char_source/DRCD_sorted.txt'
random_source = wd + '/utils/char_source/chinese_char_std_6964.txt'
fonts_folder = wd + '/utils/fonts'
# images_dst_folder = wd + '/temp_dst_folder'
images_dst_folder = 'D:/training_data/keras-chinese-ocr/train/images/006'
tmp_text_path = wd + "/tmp_text_semantic.txt"

gen_semantic = True
char_list = []
fixed_char_length = 10
max_processes = os.cpu_count()
print("max_processes = {}, will use half core".format(max_processes))
num_of_processes = max_processes // 2

lock = mp.Lock()
counter = mp.Value('i', 0)
STOP_TOKEN = 'kill'


def start_listen(q, fname):
    """ listens for messages on the q, writes to file. """
    print(fname)
    f = open(fname, mode='a', encoding='utf-8')
    while 1:
        m = q.get()
        if m == STOP_TOKEN:
            break
        try:
            f.write(str(m) + '\n')
        except:
            traceback.print_exc()

        with lock:
            if counter.value % 1000 == 0:
                f.flush()
    f.close()


def sort_labels(tmp_label_fname, label_fname):
    with open(tmp_label_fname, mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = sorted(lines)
    with open(label_fname, mode='a', encoding='utf-8') as f:
        for line in lines:
            f.write(line)


def load_char_list():
    global char_list
    if gen_semantic:
        with open(semantic_source, 'r', encoding="utf-8") as f:
            char_list = f.readlines()
    else:
        with open(random_source, 'r', encoding="utf-8") as f:
            char_list = f.read()
            char_list = char_list.split('\n')


load_char_list()


class HandWritingGenerator:

    def __init__(self, chars, font_path):
        self.chars = chars
        self.font_path = font_path
        self.font_size = 28
        self.dst_width = 280
        self.dst_height = 32

    def generate(self):
        fianl_chars = ''
        font = ImageFont.truetype(self.font_path, self.font_size)
        ttFont = TTFont(self.font_path)
        uniMap = ttFont['cmap'].tables[0].ttFont.getBestCmap()
        for i in range(len(self.chars)):
            if ord(self.chars[i]) not in uniMap.keys():
                fianl_chars += '一'
            else:
                fianl_chars += self.chars[i]
        self.chars = fianl_chars
        img_width, img_height = font.getsize(self.chars)
        w_bias = 14
        h_bias = 4
        img = Image.new('RGB', (img_width + w_bias, img_height + h_bias), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((w_bias // 2, h_bias // 2), self.chars, font=font, fill=(0, 0, 0, 0), stroke_width=0)
        img = img.resize((self.dst_width, self.dst_height), Image.BICUBIC)
        return img


def generate_img(img_index, q=None, exist_img_count=0):
    global lock, counter, char_list
    filename = str(img_index).zfill(7)

    np.random.seed()

    if gen_semantic:
        start_idx = (img_index - exist_img_count) * fixed_char_length
        try:
            chars = char_list[0][start_idx: start_idx + 10]
        except IndexError:
            return
    else:
        chars = ''
        while len(chars) < 10:
            idx = np.random.randint(0, len(char_list))
            chars += char_list[idx]

    fonts_list = glob.glob(fonts_folder + '/*')
    font_idx = np.random.randint(0, len(fonts_list))
    font_path = fonts_list[font_idx]

    hw = HandWritingGenerator(chars, font_path)
    ea = effect_utils.EffectApplier()
    ea.initialize_effect()
    text_img = hw.generate()
    variate_img = ea.do_variation(text_img)
    variate_img.save(images_dst_folder + "/006" + filename + '.jpg')

    text = u''.join(chars)
    line = "006" + filename + '.jpg ' + text
    if q is not None:
        q.put(line)


if __name__ == "__main__":
    manager = mp.Manager()
    q = manager.Queue()

    gen_images_num = len(char_list[0]) // fixed_char_length

    exist_img_list = glob.glob(images_dst_folder + "/*.jpg")
    exist_img_count = len(exist_img_list)
    start_index = exist_img_count
    end_index = start_index + gen_images_num

    start_time = time.time()
    with mp.Pool(processes=num_of_processes) as pool:
        pool.apply_async(start_listen, (q, tmp_text_path))
        pool.starmap(generate_img, zip(range(start_index, end_index), repeat(q), repeat(exist_img_count)))
        q.put(STOP_TOKEN)
        pool.close()
        pool.join()
    end_time = time.time()
    print("Finish generate data, time elapsed = {}s".format(end_time - start_time))
    text_path = images_dst_folder + "/texts.txt"
    sort_labels(tmp_text_path, text_path)