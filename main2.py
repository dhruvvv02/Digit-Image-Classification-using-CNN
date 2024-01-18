from PIL import Image
import numpy as np
image = Image.open('digits/download6.png')

image_array = np.array(image)
print("Size of image array :",image_array.size)

resize_image = image.resize((28,28))
resize_array = np.array(resize_image)
print("Size of resize image array :",resize_array.size)