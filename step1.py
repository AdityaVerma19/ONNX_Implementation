import torch
import torchvision.models as models

# 1. Load a standard pre-trained model (ResNet18)
model = models.resnet18(pretrained=True)
model.eval() # Set the model to inference/evaluation mode (CRITICAL!)

# 2. Create a "dummy" input. 
# ONNX needs this to trace the path the data takes through the model.
# ResNet takes a 3-channel image of 224x224 pixels. (Batch_size, Channels, Height, Width)
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Export the model to a .onnx file
torch.onnx.export(
    model,                      # The model we are exporting
    dummy_input,                # The dummy input
    "my_image_model.onnx",      # The name of the output file
    export_params=True,         # Store the trained weights inside the file
    opset_version=11,           # The ONNX version to use
    input_names=['input'],      # Name the input (useful for deployment)
    output_names=['output']     # Name the output
)

print("Model successfully exported to my_image_model.onnx!")