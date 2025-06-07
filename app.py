import os
import pickle
import numpy as np
import tensorflow
from tqdm import tqdm
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)
    return normalized_result

file_names = [os.path.join('images', file) for file in os.listdir('images')]
feature_list = [extract_features(file, model) for file in tqdm(file_names)]

feature_array = np.array(feature_list)
print(feature_array.shape)

pickle.dump(feature_array, open('embeddings.pkl', 'wb'))
pickle.dump(file_names, open('filenames.pkl', 'wb'))
