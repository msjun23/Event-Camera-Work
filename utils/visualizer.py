import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def tensor_to_disparity_image(tensor_data):
    assert len(tensor_data.size()) == 2
    assert (tensor_data >= 0.0).all().item()

    disparity_image = Image.fromarray(np.asarray(tensor_data * 256.0).astype(np.uint16))

    return disparity_image


def tensor_to_disparity_magma_image(tensor_data, vmax=None, mask=None, fps_text=None):
    assert len(tensor_data.size()) == 2
    assert (tensor_data >= 0.0).all().item()

    numpy_data = np.asarray(tensor_data)

    if vmax is not None:
        numpy_data = numpy_data * 255 / vmax
        numpy_data = np.clip(numpy_data, 0, 255)

    numpy_data = numpy_data.astype(np.uint8)
    numpy_data_magma = cv2.applyColorMap(numpy_data, cv2.COLORMAP_MAGMA)
    numpy_data_magma = cv2.cvtColor(numpy_data_magma, cv2.COLOR_BGR2RGB)

    if mask is not None:
        assert tensor_data.size() == mask.size()
        numpy_data_magma[~mask] = [255, 255, 255]

    disparity_image = Image.fromarray(numpy_data_magma)
    
    if fps_text is not None:
        # Draw obj for text
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(disparity_image)
        
        try:
            # Set font from file
            font = ImageFont.truetype("/root/utils/arial.ttf", 60)
        except IOError:
            # Use default font
            font = ImageFont.load_default()
        
        # Text config
        text_position = (100, 10)
        text_color = (255, 255, 0)
        
        # Write fps on pred image
        draw.text(text_position, fps_text, font=font, fill=text_color)

    return disparity_image