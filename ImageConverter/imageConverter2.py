import numpy as np
import pydicom
from PIL import Image

image = pydicom.dcmread('DCM Data/0.dcm')
image = image.pixel_array.astype(float)
rescaled_image = (np.maximum(image, 0)/image.max())*255
final_image = np.uint8(rescaled_image)
final_image = Image.fromarray(final_image)
final_image.show()
final_image.save('new image.jpg')