# Step 2: Import libraries
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from PIL import Image
import io

def extract_matrix_images_from_pdf(pdf_path, target_row="Top", target_col="In Image"):
    doc = fitz.open(pdf_path)
    page = doc[0]

    # Find the anchor "In Image : Out Image :"
    anchor_text = "In Image : Out Image :"
    anchor_rects = page.search_for(anchor_text)
    if not anchor_rects:
        ref_y = 0
    else:
        ref_y = anchor_rects[0].y1

    # Extract all image blocks
    images = []
    for img_index, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_obj = Image.open(io.BytesIO(image_bytes))
        rects = page.get_image_rects(xref)
        if rects:
            bbox = rects[0]
            images.append({
                "index": img_index,
                "image": image_obj,
                "x": bbox.x0,
                "y": bbox.y0,
                "bbox": bbox
            })

    # Detect label positions (Front, Rear, Top)
    label_positions = {}
    for label in ["Front", "Rear", "Top"]:
        found = page.search_for(label)
        if found:
            label_positions[label] = found[0]

    # Find images beside each label
    def find_images_near_label(label_rect, images, y_thresh=30):
        matched_images = []
        for img in images:
            img_y_center = img["bbox"].y0 + img["bbox"].height / 2
            label_y_center = label_rect.y0 + label_rect.height / 2
            if (
                img["x"] > label_rect.x1 and
                abs(img_y_center - label_y_center) < y_thresh
            ):
                matched_images.append(img)
        matched_images.sort(key=lambda i: i["x"])
        return matched_images[:2]

    # Build matrix
    matrix = {}
    for label, rect in label_positions.items():
        nearby_imgs = find_images_near_label(rect, images)
        matrix[label] = {}
        if len(nearby_imgs) > 0:
            matrix[label]["In Image"] = nearby_imgs[0]["image"]
        if len(nearby_imgs) > 1:
            matrix[label]["Out Image"] = nearby_imgs[1]["image"]

    # Return the selected cell as a list (for compatibility with app.py)
    selected = matrix.get(target_row, {}).get(target_col)
    return [selected] if selected else []
