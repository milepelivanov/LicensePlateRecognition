import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext, basename
from tensorflow.keras.models import model_from_json

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt
import keras_ocr


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


test_image_path = "dataset/Car.jpg"
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

save_image(LpImg[0], "detected_plate.jpg")

###################


image = cv2.imread("detected_plate.jpg")
output_folder = 'letters'


def find_contours(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    ii = cv2.imread('contour.jpg')

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:

        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
            x_cntr_list.append(
                intX)

            char_copy = np.zeros((44, 24))

            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(ii, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)
            plt.imshow(ii, cmap='gray')
            plt.title('Predict Segments')

            char = cv2.subtract(255, char)

            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)

    plt.show()

    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])
    img_res = np.array(img_res_copy)

    return img_res


def segment_characters(image):
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)

    img_gray_lp_blurred = cv2.GaussianBlur(img_gray_lp, (5, 5), 0)

    img_binary_lp = cv2.adaptiveThreshold(img_gray_lp_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                          11, 2)

    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255

    dimensions = [LP_WIDTH / 8,
                  LP_WIDTH / 1.5,
                  LP_HEIGHT / 8,
                  4 * LP_HEIGHT / 3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.title('Contour')
    plt.show()
    cv2.imwrite('contour.jpg', img_binary_lp)

    char_list = find_contours(dimensions, img_binary_lp)

    return char_list


char = segment_characters(image)

image_folder = 'letters'
shutil.rmtree(image_folder, ignore_errors=True)

os.makedirs(image_folder, exist_ok=True)

from PIL import Image

for i, char_img in enumerate(char):
    char_img_pil = Image.fromarray(char_img)
    char_img_pil = char_img_pil.convert('L')

    char_filename = os.path.join(image_folder, f'char{i + 1}.jpg')
    char_img_pil.save(char_filename)

for i in range(len(char)):
    plt.subplot(1, len(char), i + 1)
    plt.imshow(char[i], cmap='gray')
    plt.axis('off')
plt.show()

##############################


pipeline = keras_ocr.pipeline.Pipeline()

image_folder = 'letters'

image_filenames = [file for file in os.listdir(image_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
result = ""

for image_filename in image_filenames:
    image_path = os.path.join(image_folder, image_filename)

    image = keras_ocr.tools.read(image_path)

    predictions = pipeline.recognize([image])

    concatenated_text = ''.join([text for text, box in predictions[0]])

    result += concatenated_text.capitalize()

output_file_path = 'license_plate.txt'

with open(output_file_path, 'w') as output_file:
    output_file.write(str(result))
