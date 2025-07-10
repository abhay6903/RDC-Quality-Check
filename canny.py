import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from google.colab import files

# 📌 Step 1: Upload the image
uploaded = files.upload()
for fn in uploaded.keys():
    image_path = fn

# 📌 Step 2: Load and convert the image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 📌 Step 3: Crop material area
material_crop = img[0:250, 0:500]  # Adjust as needed

# 📌 Step 4: K-means segmentation
Z = material_crop.reshape((-1, 3)).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
segmented_img = centers[labels.flatten()].reshape(material_crop.shape).astype(np.uint8)

# 📌 Step 5: Convert to grayscale
gray = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2GRAY)

# 📌 Step 6: Improve contrast using histogram equalization
equalized = cv2.equalizeHist(gray)

# 📌 Step 7: Adaptive Canny (based on median intensity)
median = np.median(equalized)
lower = int(max(0, 0.66 * median))
upper = int(min(255, 1.33 * median))
edges = cv2.Canny(equalized, lower, upper)

# 📌 Step 8: Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 📌 Step 9: Draw clean contours
output = material_crop.copy()
cv2.drawContours(output, contours, -1, (0, 255, 0), 1)

# 📌 Step 10: Display results
plt.figure(figsize=(12, 6))
plt.imshow(output)
plt.title("Enhanced Wireframe")
plt.axis('off')
plt.show()

# 📌 Step 11: Optional - Edge Density %
total_pixels = edges.shape[0] * edges.shape[1]
edge_pixels = np.sum(edges > 0)
edge_coverage = (edge_pixels / total_pixels) * 100
print(f"🧮 Edge coverage: {edge_coverage:.2f}%")


# Optional: Classify based on edge density
if edge_coverage < 5:
    classification = "DE1Poor"
elif edge_coverage < 15:
    classification = "DE1Avg"
else:
    classification = "DE1Good"

print(f"📊 Texture Classification: {classification}")

