from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load your model
model = YOLO(r"C:\Users\Abhay Pandey\Desktop\pdf_app\models\material_segre.pt")

# Load a sample image (replace with a real image path)
img = Image.open(r"C:\Users\Abhay Pandey\Desktop\pdf_app\10MM_1-1-_png.rf.925c7702ee4fac79a9a167a0101d34ad.jpg").convert("RGB")
img_np = np.array(img)

# Run prediction
results = model.predict(img_np, verbose=True)

# Print the results
print("Model results:", results)
print("First result dir:", dir(results[0]))
if results:
    if hasattr(results[0], 'boxes'):
        print("Boxes:", results[0].boxes)
        for box in results[0].boxes:
            print("Box class:", getattr(box, 'cls', None))
            print("Box confidence:", getattr(box, 'conf', None))