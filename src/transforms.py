"""
Modified versions of torchvision's transforms that can be independently sampled then applied.
This allows for the same augmentation to be applied to several images (eg the temporal history)
"""

import numbers
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def sample(self, img):
        blur = random.random() < self.p
        sigma = random.random() * 1.9 + 0.1
        return (blur, sigma), self.__call__(img, (blur, sigma))

    def __call__(self, img, params):
        blur, sigma = params
        if blur:
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def sample(self, img):
        params = T.ColorJitter.get_params(
            brightness=self.brightness, hue=self.hue, saturation=self.saturation, contrast=self.contrast
        )
        return params, self.__call__(img, params)

    def __call__(self, img, params):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = params
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)
        return img


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def sample(self, img):
        region = T.RandomCrop.get_params(img, self.size)
        return region, self.__call__(img, region)

    def __call__(self, img, region):
        return F.crop(img, *region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, square: bool = False):
        self.min_size = min_size
        self.max_size = max_size
        self.square = square

    def sample(self, img):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = w if self.square else random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return region, self.__call__(img, region)

    def __call__(self, img: PIL.Image.Image, region):
        return F.crop(img, *region)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def sample(self, img):
        flip = random.random() < self.p
        return flip, self.__call__(img, flip)

    def __call__(self, img, flip):
        if flip:
            return F.hflip(img)
        return img


class RandomResize(object):
    def __init__(self, sizes, max_size=None, square=False):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
        self.square = square

    def _get_size_with_aspect_ratio(self, image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def _get_size(self, image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return self._get_size_with_aspect_ratio(image_size, size, max_size)

    def sample(self, img):
        size = random.choice(self.sizes)
        if self.square:
            size = size, size
        else:
            size = self._get_size(img.size, size, self.max_size)
        return size, self.__call__(img, size)

    def __call__(self, img, size):
        return F.resize(img, size)

class Compose(object):
    def __init__(self, transforms, p=1):
        self.transforms = transforms
        self.p = p

    def sample(self, img):
        trans_img = img
        sampled = []
        compose = random.random() < self.p
        if compose:
            for t in self.transforms:
                s, trans_img = t.sample(trans_img)
                sampled.append(s)
        return (compose, sampled), trans_img

    def __call__(self, image, params):
        compose, sampled = params
        if compose:
            assert len(self.transforms) == len(sampled)
            for t, s in zip(self.transforms, sampled):
                image = t(image, s)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
