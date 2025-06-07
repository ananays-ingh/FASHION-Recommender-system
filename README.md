# Fashion Product Image Recommender System

This project is a **content-based image recommender system** that suggests visually similar fashion products using deep learning. It uses **ResNet50** to extract embeddings from product images and provides two options for similarity search:

- **FAISS** for fast approximate nearest neighbor retrieval
- **scikit-learn NearestNeighbors** for exact brute-force comparison

The app is built with **Streamlit** to allow interactive image uploads and real-time recommendations.
