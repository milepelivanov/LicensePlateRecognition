import pytesseract as pytesseract
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext, basename
from tensorflow.keras.models import model_from_json
import keras_ocr
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import glob
import os
from PIL import Image
import pytesseract
import easyocr


def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Model Loaded successfully...")
        print("Detecting License Plate ... ")
        return model
    except Exception as e:
        print(e)


wpod_net_path = "models/wpod-net.json"
wpod_net = load_model(wpod_net_path)


def preprocess_image(image_path, resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


def get_plate(image_path, Dmax=608, Dmin=608):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _, LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor


def save_image(image, save_path):
    image_8u = np.clip(image * 255, 0, 255).astype(np.uint8)

    # Save the image
    cv2.imwrite(save_path, cv2.cvtColor(image_8u, cv2.COLOR_RGB2BGR))
    print(f"Image saved to: {save_path}")


test_image_path = "dataset/slika1.jpg"
vehicle, LpImg, cor = get_plate(test_image_path)

fig = plt.figure(figsize=(12, 6))
grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
fig.add_subplot(grid[0])
plt.axis(False)
plt.imshow(vehicle)
grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
fig.add_subplot(grid[1])
plt.axis(False)
plt.imshow(LpImg[0])
plt.show()

save_image(LpImg[0], "detected_plate.jpg")

### DO TUKA DETEKCIJA KADE SE NAOGJA REGISTRACIJATA


# net = cv2.dnn.readNet(r'C:\Users\milep\Desktop\pythonProject\EAST\frozen_east_text_detection.pb')
# image = cv2.imread('detected_plate.jpg')
# original_height, original_width = image.shape[:2]

# blob = cv2.dnn.blobFromImage(image, 1.0, (320, 320), (123.68, 116.78, 103.94), swapRB=True, crop=False)

# blob_image = blob[0, :, :, :].transpose(1, 2, 0).astype(np.uint8)
# blob_image = cv2.resize(blob_image, (blob_image.shape[1] * 2, blob_image.shape[0]))

# net.setInput(blob)

# output_layers = net.forward(['feature_fusion/Conv_7/Sigmoid'])
# for i, output_layer in enumerate(output_layers):
# print(f"Output Layer {i + 1} shape:", output_layer.shape)

# if len(output_layers[0].shape) == 4:
# scores = output_layers[0][0, 0, :, :]
# geometry = output_layers[0][0, 1:5, :, :]
# else:
# print("Unexpected output_layers[0] structure. Adjust the indexing.")

# print("Output Layers Shape:", output_layers[0].shape)
# print("Output Layers Values:", output_layers[0])


# ---------------------
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\milep\Desktop\pythonProject\Tesseract-OCR\tesseract.exe'

# image = cv2.imread('detected_plate.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# custom_config = r'--oem 3 --psm 6'

# text_boxes = pytesseract.image_to_boxes(gray, config=custom_config)

# buffer_size = 3
# output_folder = 'cropped/'

# for i, box in enumerate(text_boxes.splitlines()):
#    box = box.split()
#    x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])

#    x -= buffer_size
#    y -= buffer_size
#    w += buffer_size
#   h += buffer_size + 5

#   cropped_image = image[y:h, x:w]
#    output_path = os.path.join(output_folder, f'cropped_image_{i}.png')
#    cv2.imwrite(output_path, cropped_image)

#    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

# cv2.imshow('Result', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


### DO TUKA SEGMENTACIJA NA BUKVI

# ---------------------
# pipeline = keras_ocr.pipeline.Pipeline()
# image = keras_ocr.tools.read('detected_plate.jpg')

# predictions = pipeline.recognize([image])

# concatenated_text = ''.join([text for text, box in predictions[0]])

# fig, ax = plt.subplots(figsize=(10, 20))

# keras_ocr.tools.drawAnnotations(image=image, predictions=predictions[0], ax=ax)

# print(concatenated_text)


# plt.show()

# ---------------------
#

# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\milep\Desktop\pythonProject\Tesseract-OCR\tesseract.exe'

# img_path = 'detected_plate.jpg'
# img = cv2.imread(img_path)
#

# text = pytesseract.image_to_string(img)

# print("Extracted Text:")
# print(text)

# ---------------------


# reader = easyocr.Reader(['en'])

# result = reader.readtext('detected_plate.jpg')

# for detection in result:
#    print(detection[1])
