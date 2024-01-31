import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt
import keras_ocr
import os

pipeline = keras_ocr.pipeline.Pipeline()

image_folder = 'letters'  # Change this to the actual folder path

image_filenames = [file for file in os.listdir(image_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
result = ""

for image_filename in image_filenames:
    image_path = os.path.join(image_folder, image_filename)

    image = keras_ocr.tools.read(image_path)

    predictions = pipeline.recognize([image])

    concatenated_text = ''.join([text for text, box in predictions[0]])

    result += concatenated_text.capitalize()



print("THE LICENSE PLATE IS ", result)
