from flask import Flask, request, render_template
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load precomputed feature vectors and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the saved model from 'model.pkl'
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


# Function to save uploaded files to the 'static/uploads' directory
def save_uploaded_file(uploaded_file):
    try:
        # Secure the filename to prevent potential security issues
        filename = secure_filename(uploaded_file.filename)

        # Join the path using forward slashes
        file_path = os.path.join('static', 'uploads', filename).replace("\\", "/")

        # Save the file
        uploaded_file.save(file_path)
        print(f"Saved file at: {file_path}")
        return filename  # return just the filename
    except Exception as e:
        print(f"Error saving file: {e}")
        return None


# Function to extract features from an image using the model
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


# Function to find the nearest neighbors of the extracted features
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        uploaded_file = request.files['file']

        if uploaded_file.filename == '':
            return "No selected file"

        if uploaded_file:
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                full_path = os.path.join('static', 'uploads', file_path).replace("\\", "/")
                features = feature_extraction(full_path, model)
                indices = recommend(features, feature_list)

                # Adjust the paths to point to the correct location in the static folder
                recommended_images = [os.path.join('uploads', filenames[i]).replace("\\", "/") for i in indices[0][1:6]]

                # Print recommended image paths for debugging
                print("Recommended images:", recommended_images)

                return render_template('index.html', user_image=file_path,
                                       recommendations=recommended_images)

    return render_template('index.html', user_image=None, recommendations=None)


if __name__ == '__main__':
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')

    app.run(debug=True)
