import os
import base64
import re
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import fitz  # PyMuPDF
from ultralytics import YOLO
import numpy as np
from pdfmatrix import extract_matrix_images_from_pdf
import urllib3
from dmspdfff import fetch_and_save_base64_parallel, decode_base64_txts_to_pdf
import traceback
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
from reportlab.platypus import Image as RLImage
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB

# Load models
MODEL_PATH = r"C:\Users\Abhay Pandey\Desktop\pdf_app\models\material_segre.pt"
VGG16_FARSAND_PATH = r"C:\Users\Abhay Pandey\Desktop\pdf_app\models\vgg16_farsand.h5"
EFFICIENTNET_AGG_PATH = r"C:\Users\Abhay Pandey\Desktop\pdf_app\models\efficientnet_finetuned.h5"

model = YOLO(MODEL_PATH)
vgg16_farsand_model = load_model(VGG16_FARSAND_PATH)
efficientnet_agg_model = load_model(EFFICIENTNET_AGG_PATH)

# In-memory storage for extracted data and images
extracted_data = []  # List of dicts: {row: [...], images: [base64,...], predictions: [], quality_predictions: []}
prediction_log = []  # List to store all predictions for logging
plant_code_global = None
month_global = None
year_global = None

def preprocess_image_for_quality(img, target_size=(224, 224)):
    """Preprocess image for quality classification models"""
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

def predict_quality(img, item_name):
    """Predict quality using appropriate model based on item name"""
    img_array = preprocess_image_for_quality(img)
    
    if item_name.upper() == "FARSAND":
        predictions = vgg16_farsand_model.predict(img_array, verbose=0)
    else:  # For CA10MM and CA20MM
        predictions = efficientnet_agg_model.predict(img_array, verbose=0)
    
    # Get the class with highest probability
    quality_class = np.argmax(predictions[0])
    quality_map = {0: "DE1Good", 1: "DE1Avg", 2: "DE1Poor"}
    return quality_map[quality_class]

# Utility function for field extraction

def extract_fields_from_text(text):
    patterns = {
        "Item Name": r'Item Name\s*[:\-]?\s*([^\n\r:]+)',
        "In Date": r'In Date\s*[:\-]?\s*([\d]{2,4}[-/][\d]{1,2}[-/][\d]{1,4})',
        "In Time": r'In\s*time\s*[:\-]?\s*([\d]{1,2}:[\d]{2}(?::[\d]{2})?)',
        "Vendor Name": r'Vendor Name\s*[:\-]?\s*([^\n\r:]+)',
        "PO Number": r'PO\s*No\s*[:\-#]?\s*([A-Za-z0-9\-\/]+)'
    }
    fields = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        fields[key] = match.group(1).strip() if match else None
    return fields

def cleanup_old_files():
    """Clean up old PDF and base64 files when server starts"""
    folders = ['base64_txts', 'decoded_pdfs']
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    global extracted_data, plant_code_global, month_global, year_global
    # Clear in-memory data on each new session
    extracted_data = []
    plant_code_global = None
    month_global = None
    year_global = None
    
    # Clean up old files
    cleanup_old_files()
    
    # Read plant list from file
    with open('Plants lists.txt', 'r', encoding='utf-8') as f:
        plants = [line.strip() for line in f if line.strip()]
    
    if request.method == 'POST':
        auth_token = request.form.get('auth_token')
        plant_code = request.form.get('plant_code')
        month = request.form.get('month')  # 3-letter abbreviation
        year = request.form.get('year')
        plant_code_global = plant_code
        month_global = month
        year_global = year
        if not (auth_token and plant_code and month and year):
            return render_template('index.html', rows=[], plants=plants, pdf_count=0)
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        month_year_str = f"{month}-{year}"
        base64_folder = 'base64_txts'
        pdf_folder = 'decoded_pdfs'
        os.makedirs(base64_folder, exist_ok=True)
        os.makedirs(pdf_folder, exist_ok=True)
        import requests, json, calendar
        from datetime import datetime, timedelta
        url = "https://dms.rdc.in/rest/documentclasses/105/indexsearch"
        headers = {"Authorization": auth_token, "Content-Type": "application/json"}
        all_document_ids = []
        try:
            month_dt = datetime.strptime(month_year_str, "%b-%Y")
        except ValueError:
            return render_template('index.html', rows=[], plants=plants, pdf_count=0)
        year_int = int(year)
        month_int = month_dt.month
        start_date = datetime(year_int, month_int, 1)
        end_date = datetime(year_int, month_int, calendar.monthrange(year_int, month_int)[1])
        current_date = start_date
        while current_date <= end_date:
            in_date = current_date.strftime("%d-%b-%Y")
            payload = {
                "documentIndexes": [
                    {"indexName": "IN_DATE", "value1": in_date, "operator": 1},
                    {"indexName": "PLANT_CODE", "value1": plant_code, "operator": 1}
                ]
            }
            try:
                response = requests.post(url, headers=headers, json=payload, verify=False)
                response.raise_for_status()
                data = response.json()
                found_ids = [doc.get("documentId") for doc in data.get("documents", []) if doc.get("documentId")]
                all_document_ids.extend(found_ids)
            except Exception as e:
                pass
            current_date += timedelta(days=1)
        fetch_and_save_base64_parallel(auth_token, all_document_ids, base64_folder)
        decode_base64_txts_to_pdf(base64_folder, pdf_folder)
        import fitz
        from pdfmatrix import extract_matrix_images_from_pdf
        for pdf_file in os.listdir(pdf_folder):
            if not pdf_file.endswith('.pdf'):
                continue
            pdf_path = os.path.join(pdf_folder, pdf_file)
            doc = fitz.open(pdf_path)
            page = doc[0]
            text = page.get_text()
            fields = extract_fields_from_text(text)
            item_name = (fields.get("Item Name") or '').upper()
            if item_name not in ["CA10MM", "CA20MM", "FARSAND", "10MM", "20MM", "FARSAND"]:
                continue
            row = [None, pdf_file, 1, fields.get("Item Name"), fields.get("In Date"), fields.get("In Time"), fields.get("Vendor Name"), fields.get("PO Number")]
            images = extract_matrix_images_from_pdf(pdf_path, target_row="Top", target_col="In Image")
            image_b64s = []
            for img in images:
                # Resize image to max 512x512 before encoding
                max_size = (512, 512)
                img = img.copy()
                img.thumbnail(max_size, Image.LANCZOS)
                thumb_io = BytesIO()
                img.save(thumb_io, format="PNG")
                image_data = base64.b64encode(thumb_io.getvalue()).decode()
                image_b64s.append(image_data)
            extracted_data.append({"row": row, "images": image_b64s})
        
        # After processing PDFs, update the count
        pdf_count = len([f for f in os.listdir(pdf_folder) if f.endswith('.pdf')])
        return render_template('index.html', rows=extracted_data, plants=plants, pdf_count=pdf_count, plant_code=plant_code)
    
    return render_template('index.html', rows=[], plants=plants, pdf_count=0, plant_code=None)

@app.route('/reset', methods=['POST'])
def reset():
    global extracted_data, prediction_log, plant_code_global, month_global, year_global
    extracted_data = []
    prediction_log = []
    plant_code_global = None
    month_global = None
    year_global = None
    cleanup_old_files()
    return '', 204

@app.route('/detect', methods=['POST'])
def detect():
    image_base64 = request.form.get('image')
    pdf_file = request.form.get('pdf_file')
    if not image_base64:
        return jsonify({"labels": []})
    try:
        # Remove data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        img_bytes = BytesIO(base64.b64decode(image_base64))
        img = Image.open(img_bytes).convert("RGB")
        
        # Material detection using YOLO
        results = model.predict(img)
        predictions = []
        for box in results[0].boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls]
            predictions.append({"label": label, "confidence": conf})
        labels = [p["label"] for p in predictions]

        # Find the corresponding entry in extracted_data
        entry = None
        for e in extracted_data:
            if e['row'][1] == pdf_file:
                entry = e
                break

        if entry:
            # Store material predictions
            if 'predictions' not in entry:
                entry['predictions'] = []
            if predictions:
                entry['predictions'].append(predictions[0])

            # Get item name for quality classification
            item_name = entry['row'][3] or ''
            
            # Quality classification
            quality_prediction = predict_quality(img, item_name)
            
            # Store quality prediction
            if 'quality_predictions' not in entry:
                entry['quality_predictions'] = []
            entry['quality_predictions'].append(quality_prediction)

            return jsonify({
                "labels": labels,
                "raw_predictions": predictions,
                "quality_prediction": quality_prediction
            })
        else:
            return jsonify({"labels": labels, "raw_predictions": predictions})

    except Exception as e:
        print("Error in /detect:", str(e))
        print(traceback.format_exc())
        return jsonify({"labels": [], "raw_predictions": [], "error": str(e)}), 200

@app.route('/generate-csv', methods=['GET'])
def generate_csv():
    from io import StringIO, BytesIO  # updated here
    global plant_code_global, month_global, year_global
    if not extracted_data:
        return jsonify({"error": "No data available for report"}), 400

    output = StringIO()
    import csv

    writer = csv.writer(output)

    # Write headers
    # Metadata
    writer.writerow([])
    if plant_code_global and month_global and year_global:
        writer.writerow(['Report Details'])
        writer.writerow(['Plant ID', plant_code_global])
        writer.writerow(['Month-Year', f"{month_global}-{year_global}"])
        writer.writerow(['Generated on', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

    writer.writerow(['PDF', 'Vendor Name', 'Item Name', 'PO Number', 'IN_DATE', 'Predicted Material', 'Match', 'Quality'])

    for entry in extracted_data:
        row = entry['row']
        predictions = entry.get('predictions', [])
        pred_label = predictions[0]['label'] if predictions else 'Not Clear'
        is_match = pred_label.upper() == (row[3] or '').upper() if row[3] else False
        quality_predictions = entry.get('quality_predictions', [])
        quality = quality_predictions[0] if quality_predictions and pred_label != 'Not Clear' else ''
        writer.writerow([
          row[1],  # PDF
          row[6] or '-',  # Vendor Name
          row[3] or '-',  # Item Name
          row[7] or '-',  # PO Number
          row[4] or '-',  # IN_DATE (added here)
          pred_label,
          'Yes' if is_match else 'No',
          quality
         ])


    # Add a blank row and quality statistics
    writer.writerow([])

    material_names = ['CA10MM', 'CA20MM', 'FARSAND']
    material_totals = {name: 0 for name in material_names}
    material_quality = {name: {'Good': 0, 'Avg': 0, 'Poor': 0} for name in material_names}

    for entry in extracted_data:
        row = entry['row']
        item = (row[3] or '').upper()
        if item in material_names:
            images = entry.get('images', [])
            material_totals[item] += len(images)
            qualities = entry.get('quality_predictions', [])
            for q in qualities:
                if q == 'DE1Good':
                    material_quality[item]['Good'] += 1
                elif q == 'DE1Avg':
                    material_quality[item]['Avg'] += 1
                elif q == 'DE1Poor':
                    material_quality[item]['Poor'] += 1

    writer.writerow(['Material-wise Quality Analysis'])
    writer.writerow(['Material', 'Total Samples', 'Good %', 'Average %', 'Poor %'])

    for item in material_names:
        total = material_totals[item]
        good = material_quality[item]['Good']
        avg = material_quality[item]['Avg']
        poor = material_quality[item]['Poor']

        good_p = f"{(good / total * 100):.1f}" if total else '0'
        avg_p = f"{(avg / total * 100):.1f}" if total else '0'
        poor_p = f"{(poor / total * 100):.1f}" if total else '0'

        writer.writerow([
            item,
            total,
            good_p + '%',
            avg_p + '%',
            poor_p + '%'
        ])

    
    # Convert to bytes and send
    output.seek(0)
    csv_bytes = BytesIO(output.getvalue().encode('utf-8'))

    return send_file(
        csv_bytes,
        as_attachment=True,
        download_name=f"material_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mimetype='text/csv'
    )


if __name__ == '__main__':
    # Reset all in-memory data and clean up files on server start
    extracted_data = []
    plant_code_global = None
    month_global = None
    year_global = None
    cleanup_old_files()  # Clean up old files on server start
    app.run(debug=True)
