import pickle
import numpy as np
import tensorflow
import faiss
import matplotlib.pyplot as plt
from PIL import Image
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb'))).astype('float32')
file_names = pickle.load(open('filenames.pkl', 'rb'))

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
    return normalized_result.astype('float32')

query_path = 'sample/shirt.jpg'
query_vector = extract_features(query_path, model)

dimension = feature_list.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(feature_list)
k = 5
distances, indices = index.search(np.expand_dims(query_vector, axis=0), k)

plt.figure(figsize=(15, 5))
plt.subplot(1, 6, 1)
plt.imshow(Image.open(query_path))
plt.title("Query")
plt.axis('off')

for i, idx in enumerate(indices[0]):
    img_path = file_names[idx]
    img = Image.open(img_path)
    plt.subplot(1, 6, i + 2)
    plt.imshow(img)
    plt.title(f"Match {i + 1}")
    plt.axis('off')

plt.tight_layout()
plt.show()
