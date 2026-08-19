import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class ImageDegradation:

    @staticmethod
    def change_brightness(image):
        factor = random.uniform(0.4, 1.6)
        return ImageEnhance.Brightness(image).enhance(factor)

    @staticmethod
    def change_contrast(image):
        factor = random.uniform(0.4, 1.6)
        return ImageEnhance.Contrast(image).enhance(factor)

    @staticmethod
    def gaussian_blur(image):
        radius = random.uniform(0.5, 2.5)
        return image.filter(ImageFilter.GaussianBlur(radius))

    @staticmethod
    def gaussian_noise(image):

        img = np.array(image).astype(np.float32)

        noise = np.random.normal(
            0,
            20,
            img.shape
        )

        img = img + noise

        img = np.clip(img, 0, 255).astype(np.uint8)

        return Image.fromarray(img)

    @staticmethod
    def jpeg_compression(image):

        quality = random.randint(10, 60)

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=quality
        )

        buffer.seek(0)

        return Image.open(buffer)

    @staticmethod
    def apply(image, method):

        methods = {
            "brightness": ImageDegradation.change_brightness,
            "contrast": ImageDegradation.change_contrast,
            "blur": ImageDegradation.gaussian_blur,
            "noise": ImageDegradation.gaussian_noise,
            "jpeg": ImageDegradation.jpeg_compression,
        }

        return methods[method](image)