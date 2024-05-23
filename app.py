import io
from flask import Flask, request, jsonify, send_file
from PIL import Image
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load the PyTorch model
model = YOLO('best.pt')
classes = ['Crack' ,'Dent']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files.get('image')
    print(image)
    print("Model predicting...")
    results = model.predict(Image.open(io.BytesIO(image.read())))
    print("Model predicted")
    
    output_image_path = 'result.jpg'
    for result in results:
        result.show()  # display to screen
        result.save(filename=output_image_path)

    return send_file(output_image_path, mimetype='image/jpeg')

@app.route('/')
def index():
    return "Hello World"

if __name__ == '__main__':
    app.run()
