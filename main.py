import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
from PIL import Image

model = tf.keras.models.load_model('digit_classifier.h5')

original_image = Image.open('digits/number7.png')

resized_image = original_image.resize((28,28))          #resized image to 28x28 dimension

image_array = np.array(resized_image) / 255.0           #convert image to array and normalize

reshaped_array = image_array.reshape((1, 28, 28, 1))        #reshaping the array

#original_image.show()

pred_enc = model.predict(reshaped_array)        #enoded predicted value

pred = np.argmax(pred_enc[0])           #decoding the predicted value

print("The digit is classified as ",pred)

#print(image_array)

