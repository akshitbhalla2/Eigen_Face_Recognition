#!/usr/bin/env python
# coding: utf-8

# FACE RECOGNITION WITH PCA

## Import necessary libraries
import numpy as np
import scann
import streamlit as st
from skimage import io, transform, color
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces

# Headings
st.title("Face Recognition with Eigen Faces")
st.sidebar.title("Face Recognition with Eigen Faces")

st.markdown("This app is a dashboard for Face Recognition via Principal Component Analysis️")
st.sidebar.markdown("This app is a dashboard for Face Recognition via Principal Component Analysis️")


## Obtain the dataset
@st.cache(persist=True)
def GetData():
    faces = fetch_olivetti_faces()
    targets, data = faces["target"], faces["data"]
    return targets, data

targets, data = GetData()

## Split dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(
    data, 
    targets, 
    stratify = targets,
    test_size = 0.2,
    random_state = 0
)

## Display original images
st.subheader("Faces in database")
img_list = []
for i in range(3):
    img_list.append(x_train[i, :].reshape(64, 64))

st.image(img_list, width=185)

## Preprocess datasets
pca = PCA(n_components = 0.95)
pca.fit(x_train)

## Explore Eigenfaces
st.subheader("Eigen Faces using PCA")
img_list = []
for i in range(3):
    comp = pca.components_[i]

    num = comp - min(comp)
    den = max(comp) - min(comp)
    comp = num/den

    img_list.append(comp.reshape(64, 64))

st.image(img_list, width=185, caption=["PC1", "PC2", "PC3"])

## Transform data
x_train = pca.transform(x_train);
x_test = pca.transform(x_test);


## Classifying as recognized and not-recognized by clustering
def GetNeighbors(x_train, y_train, x_test, k):
    searcher = scann.scann_ops_pybind.builder(
        x_train, 
        k, 
        "dot_product"
    ).tree(
        num_leaves = 10, 
        num_leaves_to_search = 300, 
        training_sample_size = 500
    ).score_ah(
        2, 
        anisotropic_quantization_threshold = 0.2
    ).reorder(30).build()
    
    neighbors, distances = searcher.search_batched(x_test)
    categories = np.array(y_train)
    
    neighbors = categories[neighbors]
    
    return neighbors, distances

x_train = x_train/np.sqrt(np.sum(np.square(x_train), axis = 1, keepdims = True))

img = st.sidebar.file_uploader("Choose a .jpg image", type="jpg")
if img is not None:
    img = io.imread(img)
    img = transform.resize(img, (64, 64))
    img = color.rgb2gray(img)

    st.sidebar.image(img, width=120, caption="Test face")

    st.subheader("Uploaded Image")
    st.image(img, width=64)

    test = pca.transform(img.reshape(1, -1))
    test = test/np.sqrt(np.sum(np.square(test), axis = 1, keepdims = True))

    n, d = GetNeighbors(x_train, y_train, test, 1)

    if np.squeeze(d) < 0.6:
        st.sidebar.subheader("Face not recognized!")

    else:
        st.sidebar.subheader("Face of person: " + str(np.squeeze(n).tolist()))
