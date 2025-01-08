from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import base64
from io import BytesIO
from PIL import Image
import os
import pickle
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

model = tf.keras.models.load_model('FYPImageRecogModel.keras')
foodname= ['aloo_gobi', 'aloo_matar', 'aloo_methi', 'aloo_shimla_mirch', 'aloo_tikki', 'anarsa', 'apple', 'applelow', 'applehigh', 'applemid', 'baked_potato', 'banana', 'bananalow', 'bananahigh', 'bananasuperhigh', 'bananamid', 'beetroot', 'bell pepper', 'biryani', 'boondi', 'burger', 'butter_chicken', 'cabbage', 'capsicum', 'carrot', 'carrotlow', 'carrotmid', 'carrothigh', 'cauliflower', 'chapati', 'chena_kheeri', 'chicken_razala', 'chicken_tikka', 'chicken_tikka_masala', 'chikki', 'chilli pepper', 'corn', 'crispy_chicken', 'cucumber', 'daal_baati_churma', 'daal_puri', 'dal_makhani', 'dal_tadka', 'donut', 'doodhpak', 'dum_aloo', 'eggplant', 'fries', 'gajar_ka_halwa', 'garlic', 'ghevar', 'ginger', 'grapes', 'gulab_jamun', 'hot_dog', 'jalebi', 'jalepeno', 'kachori', 'kadai_paneer', 'kadhi_pakoda', 'kajjikaya', 'kalakand', 'karela_bharta', 'kiwi', 'kofta', 'lassi', 'lemon', 'lettuce', 'litti_chokha', 'lyangcha', 'makki_di_roti_sarson_da_saag', 'malapua', 'mango', 'misi_roti', 'misti_doi', 'modak', 'naan', 'navrattan_korma', 'onion', 'orange', 'palak_paneer', 'paneer_butter_masala', 'paprika', 'pear', 'peas', 'phirni', 'pineapple', 'pizza', 'poha', 'pomegranate', 'poornalu', 'pootharekulu', 'potato', 'raddish', 'rasgulla', 'sandwich', 'sheera', 'shrikhand', 'sohan_papdi', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'taco', 'taquito', 'tomato', 'tomatolow', 'tomatohigh', 'tomatomid', 'turnip', 'unni_appam', 'watermelon']

# 'tomato(1-5)', 'tomato(5-10)', 'tomato(10-15)'           (1-5)/(1-2) = low 
# 'carrot(1-2)', 'carrot(3-4)', 'carrot(5-6)'               (5-10)/(3-4) = mid 
# 'apple(1-5)', 'apple(5-10)',  'apple(10-14)',             (10-15)/(10-14)/(5-6) = high
# 'banana(1-5)', 'banana(5-10)', 'banana(10-15)', 'banana(15-20)', (15-20)  = superhigh

def uri_to_image(uri):
    # Remove the 'data:image/jpeg;base64,' prefix if it exists
    if uri.startswith('data:image/jpeg;base64,'):
        uri = uri[len('data:image/jpeg;base64,'):]

    # Decode the Base64 string
    image_data = base64.b64decode(uri)

    # Convert the image data to a PIL Image
    image = Image.open(BytesIO(image_data))

    return image

def save_image_from_uri(uri, filename):
    # Convert URI to Image
    image = uri_to_image(uri)

    # Save the image as a JPG file
    image.save(filename, format='JPEG')
    image.save('upload/saved.jpg', format='JPEG')

def remove_saved_image(filename):
    # Check if the file exists before attempting to remove it
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print(f"File '{filename}' not found.")

def preprocess_image(path):
    try:
        img = cv.imread(path)

        # Check if the image is loaded successfully
        if img is None:
            raise Exception("Error: Unable to load image.")

        # Check the original image shape
        print(f"Original image shape: {img.shape}")

        new_arr = cv.resize(img, (64, 64))

        # Check the resized image shape
        print(f"Resized image shape: {new_arr.shape}")

        new_arr = np.array(new_arr)
        new_arr = new_arr.reshape(1, 64, 64, 3)
        
        return new_arr

    except Exception as e:
        print(f"Image processing error: {str(e)}")
        return None

@app.route('/')
def index():
    hi = "hello world"
    return jsonify({'show':hi})
    
@app.route('/mlpredict', methods=['POST'])
def MLPredict():
    try:
        data = request.get_json()
        val1 = data['input_1']
        val2 = data['input_2']
        val3 = data['input_3']
        val4 = data['input_4']
        val5 = data['input_5']
    except KeyError as e:
        return jsonify({"message": "problem in input"}), 400
    # ------------------------------------------------------------------
    try:
        rest = joblib.load('FYPworkout.joblib')
    except KeyError as e:
        return jsonify({"message": "model is not being load"}), 400
    # ------------------------------------------------------------------
    try:
        test = np.array([[val1,val2,val3,val4,val5]])
        arr = np.array(["4day1time","6day1time","4day2time","6day2time","bodyweight","notWork"])
    except KeyError as e:
        return jsonify({"message": "there is an error in makeing array of test or arr"}), 400
    # ------------------------------------------------------------------
    
    predi = rest.predict(np.array([[val1,val2,val3,val4,val5]]))
    result = str(arr[predi[0]])
    return jsonify({'show':"it pass all thing","predict":result})

@app.route('/cnn', methods=['POST'])
def CNN():
    try:
        data = request.get_json()
        image = data['imageURI']
        filename = 'saved_image.jpg'
        save_image_from_uri(image, filename)

        img_array = preprocess_image(filename)

        # Check if there was an issue with image processing
        if img_array is None:
            return jsonify({'show': "Image processing error. Please check the uploaded image."})
        
        prediction = model.predict(img_array)
        result=foodname[prediction.argmax()]
        remove_saved_image(filename)
    except KeyError as e:
        return jsonify({'show':"there is an input error"})
    
    return jsonify({'show':"everything we got it",'result':result})

if __name__ == '__main__':
    app.run(debug=True)

