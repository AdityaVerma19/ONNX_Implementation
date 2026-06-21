
import onnxruntime as ort
import numpy as np
import requests
import urllib.request
from PIL import Image

# ==========================================
# SETUP: Get the model and the labels
# ==========================================

# 1. Download the official list of the 1,000 words the model knows (ImageNet classes)
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = urllib.request.urlopen(url).read().decode("utf-8").split("\n")

# 2. Load your ONNX model into the Runtime Session
# (Assuming you exported 'my_image_model.onnx' from the previous example)
session = ort.InferenceSession("my_image_model.onnx")

# ==========================================
# PHASE 1: Pre-processing (Image -> Numbers)
# ==========================================

# Let's create a fake "image" filled with random noise just for testing.
# In real life, you would load a real image using the PIL library here.
# Models expect images in the format: (Batch_Size, Channels, Height, Width)
# Channels = 3 (Red, Green, Blue). Height/Width = 224 pixels.
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# ==========================================
# PHASE 2: Inference (Running the ONNX model)
# ==========================================

# We pass the input_data into the model.
# 'input' is the name we gave the starting point of the model when we exported it.
raw_outputs = session.run(None, {'input': input_data})

# The model returns a list of outputs. We only care about the first one.
predictions = raw_outputs[0] 

# Right now, 'predictions' is an array of 1,000 numbers. It looks like:
# [[-0.45, 1.23, -3.44, ... 0.88]] 
# This is what confused you! Let's fix it in Phase 3.

# ==========================================
# PHASE 3: Post-processing (Numbers -> English)
# ==========================================

# 1. Strip away the batch size (we only passed 1 image, so we just want the list of 1,000 scores)
scores = predictions[0]

# 2. Find the INDEX (the position in the list) of the highest score.
# np.argmax looks at all 1,000 numbers and says, "The biggest number is at position #207"
winning_index = np.argmax(scores)

# 3. Look up what word corresponds to that index in our labels list
winning_score = scores[winning_index]
winning_label = labels[winning_index]

# 4. Print the final, human-readable result!
print(f"The model thinks this is a: {winning_label}")
print(f"Raw confidence score: {winning_score}")