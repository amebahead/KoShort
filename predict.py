# ================================
# predict.py
# ================================
import torch
import matplotlib.pyplot as plt
from model import KeypointRegressor
from dataset import CatKeypointDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = KeypointRegressor().to(device)
model.load_state_dict(torch.load("keypoint_model.pth", map_location=device))
model.eval()

dataset = CatKeypointDataset("data/images", "data/labels")
image, _ = dataset[0]
input_tensor = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)[0].cpu().reshape(-1, 2)

img_np = image.permute(1, 2, 0).numpy()
h, w = img_np.shape[:2]
plt.imshow(img_np)
for x, y in output:
    plt.scatter(x * w, y * h, c='r', s=10)
plt.title("Predicted keypoints")
plt.show()