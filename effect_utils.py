import math_utils
import effects
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from random import shuffle


class EffectSuper:
    def __init__(self):
        self.dst_width = 280
        self.dst_height = 32

    def apply_effect(self, pil_img, value):
        pass

    def get_type(self):
        pass

    def collect_context(self):
        EffectApplier().register(self.get_type(), self)


class Effect_none(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        return pil_img

    def get_type(self):
        return 'none'


# class Effect_blur(EffectSuper):
#     def __init__(self):
#         super().__init__()
#
#     def apply_effect(self, pil_img, value):
#         output_img = pil_img.filter(ImageFilter.BLUR)
#         return output_img
#
#     def get_type(self):
#         return 'blur'


class Effect_contour(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        output_img = pil_img.filter(ImageFilter.CONTOUR)
        return output_img

    def get_type(self):
        return 'contour'


class Effect_detail(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        output_img = pil_img.filter(ImageFilter.DETAIL)
        return output_img

    def get_type(self):
        return 'detail'


class Effect_edge_enhance(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        output_img = pil_img.filter(ImageFilter.EDGE_ENHANCE)
        return output_img

    def get_type(self):
        return 'edge_enhance'


class Effect_edge_enhance_more(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        output_img = pil_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        return output_img

    def get_type(self):
        return 'edge_enhance_more'


class Effect_emboss(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        output_img = pil_img.filter(ImageFilter.EMBOSS)
        return output_img

    def get_type(self):
        return 'emboss'


class Effect_find_edges(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        output_img = pil_img.filter(ImageFilter.FIND_EDGES)
        return output_img

    def get_type(self):
        return 'find_edges'


class Effect_sharpen(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        output_img = pil_img.filter(ImageFilter.SHARPEN)
        return output_img

    def get_type(self):
        return 'sharpen'


# class Effect_smooth(EffectSuper):
#     def __init__(self):
#         super().__init__()
#
#     def apply_effect(self, pil_img, value):
#         output_img = pil_img.filter(ImageFilter.SMOOTH)
#         return output_img
#
#     def get_type(self):
#         return 'smooth'


# class Effect_smooth_more(EffectSuper):
#     def __init__(self):
#         super().__init__()
#
#     def apply_effect(self, pil_img, value):
#         output_img = pil_img.filter(ImageFilter.SMOOTH_MORE)
#         return output_img
#
#     def get_type(self):
#         return 'smooth_more'


class Effect_sharpness(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        enhancer = ImageEnhance.Sharpness(pil_img)
        valuex2 = value * 2
        v = np.random.rand() * valuex2 - value
        factor = 1.0 + v / 10
        output_img = enhancer.enhance(factor)
        return output_img

    def get_type(self):
        return 'sharpness'


class Effect_contrast(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        enhancer = ImageEnhance.Contrast(pil_img)
        valuex2 = value * 2
        v = np.random.rand() * valuex2 - value
        factor = 1.0 + v / 10
        output_img = enhancer.enhance(factor)
        return output_img

    def get_type(self):
        return 'contrast'


class Effect_brightness(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        enhancer = ImageEnhance.Brightness(pil_img)
        valuex2 = value * 2
        v = np.random.rand() * valuex2 - value
        factor = 1.0 + v / 10
        output_img = enhancer.enhance(factor)
        return output_img

    def get_type(self):
        return 'brightness'


class Effect_rotate_x(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        np_img = np.array(pil_img)
        valuex2 = value * 2
        x = np.random.rand() * valuex2 - value
        transformer = math_utils.PerspectiveTransform(x, 0, 0, scale=1.0, fovy=50)
        dst_img = transformer.transform_image(np_img)
        pil_dst_img = Image.fromarray(dst_img)
        return pil_dst_img

    def get_type(self):
        return 'rotatex'


class Effect_rotate_y(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        np_img = np.array(pil_img)
        valuex2 = value * 2
        y = np.random.rand() * valuex2 - value
        transformer = math_utils.PerspectiveTransform(0, y, 0, scale=1.0, fovy=50)
        dst_img = transformer.transform_image(np_img)
        pil_dst_img = Image.fromarray(dst_img)
        return pil_dst_img

    def get_type(self):
        return 'rotatey'


class Effect_rotate_z(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        np_img = np.array(pil_img)
        valuex2 = value * 2
        z = np.random.rand() * valuex2 - value
        transformer = math_utils.PerspectiveTransform(0, 0, z, scale=1.0, fovy=50)
        dst_img = transformer.transform_image(np_img)
        pil_dst_img = Image.fromarray(dst_img)
        return pil_dst_img

    def get_type(self):
        return 'rotatez'


class Effect_rotate_xyz(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        np_img = np.array(pil_img)
        valuex2 = value * 2
        valuex3 = value * 3
        valuex6 = value * 6
        x = np.random.rand() * valuex6 - valuex3
        y = np.random.rand() * valuex6 - valuex3
        z = np.random.rand() * valuex2 - value
        transformer = math_utils.PerspectiveTransform(x, y, z, scale=1.0, fovy=50)
        dst_img = transformer.transform_image(np_img)
        pil_dst_img = Image.fromarray(dst_img)
        return pil_dst_img

    def get_type(self):
        return 'rotatexyz'


class Effect_shifth(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        valuex2 = value * 2
        shift = int(np.random.rand() * valuex2 - value)
        if shift > 0:
            blank_img = Image.new('RGB', (self.dst_width + shift, self.dst_height), (255, 255, 255))
            blank_img.paste(pil_img, (shift, 0))
            area = (0, 0, self.dst_width, self.dst_height)
            crop_img = blank_img.crop(area)
            return crop_img
        else:
            area = (-shift, 0, self.dst_width, self.dst_height)
            crop_img = pil_img.crop(area)
            blank_img = Image.new('RGB', (self.dst_width, self.dst_height), (255, 255, 255))
            blank_img.paste(crop_img, (0, 0))
            return blank_img

    def get_type(self):
        return 'shifth'


class Effect_shiftv(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        valuex2 = value * 2
        shift = int(np.random.rand() * valuex2 - value)
        if shift > 0:
            blank_img = Image.new('RGB', (self.dst_width, self.dst_height + shift), (255, 255, 255))
            blank_img.paste(pil_img, (0, shift))
            area = (0, 0, self.dst_width, self.dst_height)
            crop_img = blank_img.crop(area)
            return crop_img
        else:
            area = (0, -shift, self.dst_width, self.dst_height)
            crop_img = pil_img.crop(area)
            blank_img = Image.new('RGB', (self.dst_width, self.dst_height), (255, 255, 255))
            blank_img.paste(crop_img, (0, 0))
            return blank_img

    def get_type(self):
        return 'shiftv'


class Effect_shifthv(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        valuex2 = value * 2
        sh = int(np.random.rand() * valuex2 - value)
        sv = int(np.random.rand() * valuex2 - value)

        if sh > 0:
            blank_img = Image.new('RGB', (self.dst_width + sh, self.dst_height), (255, 255, 255))
            blank_img.paste(pil_img, (sh, 0))
            area = (0, 0, self.dst_width, self.dst_height)
            h_crop_img = blank_img.crop(area)

            if sv > 0:
                blank_img = Image.new('RGB', (self.dst_width, self.dst_height + sv), (255, 255, 255))
                blank_img.paste(h_crop_img, (0, sv))
                area = (0, 0, self.dst_width, self.dst_height)
                crop_img = blank_img.crop(area)
                return crop_img
            else:
                area = (0, -sv, self.dst_width, self.dst_height)
                crop_img = h_crop_img.crop(area)
                blank_img = Image.new('RGB', (self.dst_width, self.dst_height), (255, 255, 255))
                blank_img.paste(crop_img, (0, 0))
                return blank_img
        else:
            area = (-sh, 0, self.dst_width, self.dst_height)
            crop_img = pil_img.crop(area)
            h_blank_img = Image.new('RGB', (self.dst_width, self.dst_height), (255, 255, 255))
            h_blank_img.paste(crop_img, (0, 0))

            if sv > 0:
                blank_img = Image.new('RGB', (self.dst_width, self.dst_height + sv), (255, 255, 255))
                blank_img.paste(h_blank_img, (0, sv))
                area = (0, 0, self.dst_width, self.dst_height)
                crop_img = blank_img.crop(area)
                return crop_img
            else:
                area = (0, -sv, self.dst_width, self.dst_height)
                crop_img = h_blank_img.crop(area)
                blank_img = Image.new('RGB', (self.dst_width, self.dst_height), (255, 255, 255))
                blank_img.paste(crop_img, (0, 0))
                return blank_img

    def get_type(self):
        return 'shifthv'


class Effect_zoom(EffectSuper):
    def __init__(self):
        super().__init__()

    def apply_effect(self, pil_img, value):
        ori_width, ori_height = pil_img.size
        valuex2 = value * 2
        s = np.random.rand() * valuex2 - value
        scale = 1 + s / 100
        new_width = int(scale * ori_width)
        new_height = int(scale * ori_height)

        if s > 1.0:
            offset_x = int((new_width - ori_width) // 2)
            offset_y = int((new_height - ori_height) // 2)
            zoom_img = pil_img.resize((new_width, new_height), Image.ANTIALIAS)
            area = (offset_x, offset_y, offset_x + self.dst_width, offset_y + self.dst_height)
            zoom_img = zoom_img.crop(area)
            return zoom_img
        else:
            offset_x = int((ori_width - new_width) // 2)
            offset_y = int((ori_height - new_height) // 2)
            blank_img = Image.new('RGB', (self.dst_width, self.dst_height), (255, 255, 255))
            zoom_img = pil_img.resize((new_width, new_height), Image.ANTIALIAS)
            blank_img.paste(zoom_img, (offset_x, offset_y))
            return blank_img

    def get_type(self):
        return 'zoom'


class EffectApplier:

    effects = {}

    def __init__(self):
        self.effect_list1 = effects.effect_list1
        self.effect_list2 = effects.effect_list2
        self.effect_list3 = effects.effect_list3

    def initialize_effect(self):
        Effect_none().collect_context()
        Effect_edge_enhance_more().collect_context()
        Effect_edge_enhance().collect_context()
        Effect_find_edges().collect_context()
        # Effect_smooth().collect_context()
        # Effect_smooth_more().collect_context()
        Effect_sharpness().collect_context()
        Effect_sharpen().collect_context()
        Effect_brightness().collect_context()
        # Effect_blur().collect_context()
        Effect_rotate_x().collect_context()
        Effect_rotate_y().collect_context()
        Effect_rotate_z().collect_context()
        Effect_rotate_xyz().collect_context()
        Effect_shifth().collect_context()
        Effect_shiftv().collect_context()
        Effect_shifthv().collect_context()
        Effect_contour().collect_context()
        Effect_contrast().collect_context()
        Effect_detail().collect_context()
        Effect_emboss().collect_context()
        Effect_zoom().collect_context()

    def get_effect_by_type(self, effect_type):
        return self.effects[effect_type]

    def register(self, effect_type, effect):
        if effect_type == "":
            raise Exception("strategyType can't be null")
        self.effects[effect_type] = effect

    def do_variation(self, pil_img, filename):
        shuffle(self.effect_list1)
        effect1 = self.effect_list1[:1]
        tmp_img = self.apply_effect(pil_img, effect1[0])

        shuffle(self.effect_list2)
        effect2 = self.effect_list2[:1]
        tmp_img = self.apply_effect(tmp_img, effect2[0])

        shuffle(self.effect_list3)
        effect3 = self.effect_list3[:1]
        dst_img = self.apply_effect(tmp_img, effect3[0])
        # print(filename, effect1[0], effect2[0], effect3[0])
        return dst_img

    def apply_effect(self, pil_img, effect_type):
        pos = effect_type.find("+-")
        effect_name = effect_type[:pos]
        value = int(effect_type[pos + 2:])
        effect = self.get_effect_by_type(effect_name)
        dst_img = effect.apply_effect(pil_img, value)
        return dst_img

