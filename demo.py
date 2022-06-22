from collections import namedtuple
from itertools import groupby
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core

# Directories where data will be placed
model_folder = "model"
data_folder = "data"
charlist_folder = f"{data_folder}/charlists"
# Precision used by model
precision = "FP16"

# To group files, you have to define the collection. In this case, you can use `namedtuple`.
Language = namedtuple(
    typename="Language", field_names=["model_name", "charlist_name", "demo_image_name"]
)
chinese_files = Language(
    model_name="handwritten-simplified-chinese-recognition-0001",
    charlist_name="chinese_charlist.txt",
    demo_image_name="handwritten_chinese_test.jpg",
)
japanese_files = Language(
    model_name="handwritten-japanese-recognition-0001",
    charlist_name="japanese_charlist.txt",
    demo_image_name="handwritten_japanese_test.png",
)

print("1 - Choose a language model to download, either Chinese or Japanese.")
# Select language by using either language='chinese' or language='japanese'
language = "chinese"
languages = {"chinese": chinese_files, "japanese": japanese_files}
selected_language = languages.get(language)

# Download the model
path_to_model_weights = Path(f'{model_folder}/intel/{selected_language.model_name}/{precision}/{selected_language.model_name}.bin')
path_to_model_weights = Path(f'xml/FP16/handwritten-simplified-chinese-recognition-0001.bin')
if not path_to_model_weights.is_file():
    download_command = f'omz_downloader --name {selected_language.model_name} --output_dir {model_folder} --precision {precision}'
    print(download_command)
    # ! $download_command
else:
    print("model has been downloaded.")

print("2 - Load the model, and print its input and output")
ie = Core()
path_to_model = path_to_model_weights.with_suffix(".xml")
model = ie.read_model(model=path_to_model)
# Select Device Name
compiled_model = ie.compile_model(model=model, device_name="CPU")
recognition_output_layer = compiled_model.output(0)
recognition_input_layer = compiled_model.input(0)
print("- model input shape: {}".format(recognition_input_layer))
print("- model output shape: {}".format(recognition_output_layer))



print("3 - load image to test.")
# Read file name of demo file based on the selected model

# file_name = selected_language.demo_image_name
file_name = "000003.jpg"
# Text detection models expects an image in grayscale format
# IMPORTANT!!! This model allows to read only one line at time
# Read image
image = cv2.imread(filename=f"{data_folder}/{file_name}", flags=cv2.IMREAD_GRAYSCALE)
# Fetch shape
image_height, _ = image.shape
print("- Original image shape: {}".format(image.shape))
print("- Image scale needs to be reshaped into: {}".format(recognition_input_layer.shape))
# B,C,H,W = batch size, number of channels, height, width
_, _, H, W = recognition_input_layer.shape
print("- We need to first resize image then add paddings in order to align with model input size.")
# Calculate scale ratio between input shape height and image height to resize image
scale_ratio = H / image_height
# Resize image to expected input sizes
resized_image = cv2.resize(
    image, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA
)
# Pad image to match input size, without changing aspect ratio
resized_image = np.pad(
    resized_image, ((0, 0), (0, W - resized_image.shape[1])), mode="edge"
)
# Reshape to network the input shape
input_image = resized_image[None, None, :, :]

## Visualise Input Image
plt.figure()
plt.axis("off")
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.figure(figsize=(20, 1))
plt.axis("off")
plt.imshow(resized_image, cmap="gray", vmin=0, vmax=255)
plt.show()

print("4 - Prepare Charlist, which is a ground truth list which we could match with our inference results.")
# Get dictionary to encode output, based on model documentation
used_charlist = selected_language.charlist_name
# With both models, there should be blank symbol added at index 0 of each charlist
blank_char = "~"
with open(f"{charlist_folder}/{used_charlist}", "r", encoding="utf-8") as charlist:
    letters = blank_char + "".join(line.strip() for line in charlist)

# Run inference on the model
predictions = compiled_model([input_image])[recognition_output_layer]
print("5 - Model Inference. Prediction results shape: {}".format(predictions.shape))
# Remove batch dimension
predictions = np.squeeze(predictions)
print("- We first squeeze the inference result into shape: {}".format(predictions.shape))
# Run argmax to pick the symbols with the highest probability
predictions_indexes = np.argmax(predictions, axis=1)
# Use groupby to remove concurrent letters, as required by CTC greedy decoding
output_text_indexes = list(groupby(predictions_indexes))
# Remove grouper objects
output_text_indexes, _ = np.transpose(output_text_indexes, (1, 0))
print("- We find out the highest probability character, and remove concurrent letters and grouper objects into shape: {}".format(output_text_indexes.shape))
# Remove blank symbols
output_text_indexes = output_text_indexes[output_text_indexes != 0]
print("- We remove blank symbolsa into shape: {}".format(output_text_indexes.shape))
# Assign letters to indexes from output array
output_text = [letters[letter_index] for letter_index in output_text_indexes]
print("- Final results: {}".format(output_text))
# Print Output
plt.figure(figsize=(20, 1))
plt.axis("off")
plt.imshow(resized_image, cmap="gray", vmin=0, vmax=255)
