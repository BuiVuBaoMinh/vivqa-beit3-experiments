import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
from pali import PaLI3B

# Load image from URL
url = "https://upload.wikimedia.org/wikipedia/commons/9/99/Sample_User_Icon.png"
image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")

# Sample text input and label
input_text = ["What is in the image?"]
label_text = ["A user icon."]

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PaLI3B().to(device)
model.eval()  # or model.train() if testing training step

with open("/root/projects/exp1/src/PaLI/pali_architecture.txt", "w") as f:
    for name, param in model.named_parameters():
        if not param.requires_grad:
            # print(f"Frozen parameter block: {name}")
            f.write(f"Frozen parameter block: {name}\n")
        else:
            # print(f"Trainable parameter block: {name}")
            f.write(f"Trainable parameter block: {name}\n")

# Forward pass
with torch.no_grad():
    output = model(images=image, texts=input_text, labels=label_text)

print("Loss:", output.loss.item())

# Inference (no labels)
with torch.no_grad():
    preds = model.generate(images=image, texts=input_text)
    print("Prediction:", preds[0])
