import onnxruntime as ort
import numpy as np

# 1. Load the ONNX file into the Runtime Session
session = ort.InferenceSession("my_image_model.onnx")

# 2. Create some fake data (as a NumPy array, no PyTorch required!)
# We use float32 because neural networks generally expect 32-bit floats
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 3. Run the model!
# We pass a dictionary matching the input name we defined earlier ('input')
outputs = session.run(None, {'input': input_data})

# 4. View the results
predicted_classes = outputs[0]
print(f"Model prediction shape: {predicted_classes.shape}") 
# Output will be (1, 1000) because ResNet predicts across 1000 classes.