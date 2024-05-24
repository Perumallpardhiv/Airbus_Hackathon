import io
from flask import Flask, request, jsonify, send_file
from PIL import Image
from ultralytics import YOLO
import base64

# Initialize Flask app
app = Flask(__name__)

# Load the PyTorch model
model = YOLO('best.pt')
classes = ['Crack' ,'Dent']

def image_to_base64(image_data):
    img = Image.fromarray(image_data)
    with io.BytesIO() as buffer:
        img.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_str

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files.get('image')
    # print(image)
    print("Model predicting...")
    results = model.predict(Image.open(io.BytesIO(image.read())))
    print("Model predicted")
    
    # output_image_path = 'result.jpg'
    print(results)
    print(results.__len__() , "results found")
    # for result in results:
        # result.show()  # display to screen
        # result.save(filename=output_image_path)
    
    base64_str = image_to_base64(results[0].orig_img)
    print(base64_str)
    print("Image converted to base64")
    # print(results)
    # return jsonify({'image': base64_str})
    return jsonify({'base64': base64_str, 'success': "success"})
    # return send_file(output_image_path, mimetype='image/jpeg')

@app.route('/')
def index():
    return "Hello World"

if __name__ == '__main__':
    app.run()
