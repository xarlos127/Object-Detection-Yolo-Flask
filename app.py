from flask import Flask, request, render_template, jsonify, send_from_directory
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import io
import base64
from tinydb import TinyDB, Query
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
model = YOLO("yolov8n.pt")  # Use the YOLOv8 model
db = TinyDB('db.json')  # Initialize the database
os.makedirs('images', exist_ok=True)

def process_image(file):
    """Load, resize, and prepare the image for detection."""
    img = Image.open(file).convert("RGB")
    img = cv2.resize(np.array(img), (512, 512))
    return img

def annotate_image(image, results):
    """Annotate the image with bounding boxes and labels."""
    detected_objects = []
    for result in results:
        for box in result.boxes:
            detected_objects.append(result.names[int(box.cls[0])])
            cv2.rectangle(image, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(image, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return image, detected_objects

def encode_image(image):
    """Convert the annotated image to a base64-encoded string."""
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the 'images' directory."""
    return send_from_directory('images', filename)


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    """Handle image upload and save to local folder with metadata."""
    file = request.files.get('image')
    # Secure the filename and save the file
    file_path = os.path.join('images', file.filename)
    file.save(file_path)

    # Save metadata to the database
    db.insert({
        'file_name': file.filename,
        'image_keywords': request.form.get('keywords', ''),
    })

    return render_template('index.html')

@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    """Handle image upload, detection, and response."""
    file = request.files.get('image')
    # Secure the filename and save the file
    file_path = os.path.join('images', file.filename)
    file.save(file_path)

    img = process_image(file)
    results = model.predict(img, conf=0.5)
    annotated_img, detected_objects = annotate_image(img, results)
    image_base64 = encode_image(annotated_img)

    # Save metadata to the database
    db.insert({
        'file_name': file.filename,
        'image_keywords': detected_objects,
    })


    return jsonify({
        "image": image_base64,
        "detected_objects": detected_objects
    })
    
@app.route('/search/', methods=['POST']) 
def search_images():
    """Search for images in the database based on Objectes detected in image using YOLO."""
    file = request.files.get('search-image')
    img = process_image(file)
    results = model.predict(img, conf=0.5)
    annotated_img, detected_objects = annotate_image(img, results)
    query = Query()
    # Run detection on the uploaded image
    # Search for images with detected objects in the database
    images = db.search(query.image_keywords.any(detected_objects)) # this should be a list of images that match the detected objects
    return jsonify({"images": [img['file_name'] for img in images]})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
