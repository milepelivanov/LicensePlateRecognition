import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt
import keras_ocr
import os

# Initialize the pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# Folder containing the images
image_folder = 'letters'  # Change this to the actual folder path

# List all image files in the folder
image_filenames = [file for file in os.listdir(image_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
result = ""

for image_filename in image_filenames:
    # Create the full path for the image
    image_path = os.path.join(image_folder, image_filename)

    # Read the image
    image = keras_ocr.tools.read(image_path)

    # Recognize text in the image
    predictions = pipeline.recognize([image])

    # Concatenate recognized text from each box
    concatenated_text = ''.join([text for text, box in predictions[0]])

    result += concatenated_text.capitalize()

    # Show the plotted image



print("THE LICENSE PLATE IS ", result)
