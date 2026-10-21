"""Lossless RGB transport without retaining every decoded rollout frame."""
import io
from PIL import Image


def encode_rgb(rgb):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format='PNG')
    return buffer.getvalue()


class EncodedRGB:
    """Image-compatible handle; conversion decodes only for the current caller."""
    def __init__(self, data):
        self.data = data

    def convert(self, mode):
        with Image.open(io.BytesIO(self.data)) as image:
            return image.convert(mode)

    def save(self, destination, **kwargs):
        with Image.open(io.BytesIO(self.data)) as image:
            image.save(destination, **kwargs)


def frame_image(frame):
    return EncodedRGB(frame) if isinstance(frame, bytes) else Image.fromarray(frame)
