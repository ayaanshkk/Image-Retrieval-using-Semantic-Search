import streamlit as st
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

# Initialize the feature extractor
fe = FeatureExtractor()

# Read image features
vgg16_features = []
clip_features = []
img_paths = []

# Load VGG16 features
for feature_path in Path("./static/feature/vgg16").glob("*.npy"):
    vgg16_features.append(np.load(feature_path))
    img_paths.append("static/img/" + feature_path.stem + ".jpg")
vgg16_features = np.array(vgg16_features)

n_samples, n_features = vgg16_features.shape
n_components = min(n_samples, n_features, 256)  
pca = PCA(n_components=n_components)
vgg16_features = pca.fit_transform(vgg16_features)


vgg16_features /= np.linalg.norm(vgg16_features, axis=1, keepdims=True)

# Load CLIP features
for feature_path in Path("./static/feature/clip").glob("*.npy"):
    clip_features.append(np.load(feature_path))
clip_features = np.array(clip_features)
clip_features /= np.linalg.norm(clip_features, axis=1, keepdims=True)  # Normalize features

# Function to search images based on VGG16 features
def search_vgg16(query):
    query = pca.transform(query.reshape(1, -1))  # Apply PCA to query as well
    query /= np.linalg.norm(query)  # Normalize query
    dists = np.linalg.norm(vgg16_features - query, axis=1)
    ids = np.argsort(dists)[:30]
    scores = [(dists[id], img_paths[id]) for id in ids]
    return scores

# Function to search images based on CLIP features
def search_clip(query):
    similarities = np.dot(clip_features, query.T).flatten()
    ids = np.argsort(similarities)[::-1][:30]
    scores = [(similarities[id], img_paths[id]) for id in ids]
    return scores

# Function to filter results based on a similarity threshold
def filter_results(scores, threshold=0.2):
    filtered_scores = [score for score in scores if score[0] > threshold]
    return filtered_scores

def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Select your choice", ["Home", "Image Retrieval Engine"])

    if choice == "Home":
        image = Image.open(r'C:\Users\ayashaik1\Image generator project\logo.png.png')
        st.image(image, width=100)
        st.title('Image Retrieval Application using Semantic Search')

    elif choice == "Image Retrieval Engine":
        st.title("Image Retrieval Engine")

        # Image upload section
        image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if image:
            st.image(image, caption='Query Image', use_column_width=True)
            img = Image.open(image).convert('RGB')
            uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + image.name
            img.save(uploaded_img_path)

            # Extract features and find similar images
            query = fe.extract_image_vgg16(img)
            scores = search_vgg16(query)

            st.write("Results:")
            columns = st.columns(2)
            for i in range(len(scores)):
                with columns[i % 2]:
                    st.image(scores[i][1], caption=f'Score: {scores[i][0]}', use_column_width=True)

        # Text input section
        text = st.text_input("Enter search text")
        if text:
            query = fe.extract_text(text)
            scores = search_clip(query)
            scores = filter_results(scores)

            st.write("Results:")
            columns = st.columns(2)
            for i in range(len(scores)):
                with columns[i % 2]:
                    st.image(scores[i][1], caption=f'Score: {scores[i][0]}', use_column_width=True)

if __name__ == "__main__":
    main()
