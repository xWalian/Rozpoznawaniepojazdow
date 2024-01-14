import numpy as np
from skimage.filters import threshold_otsu
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from skimage import io, color
import joblib
import os

def load_custom_images(directory, target_size=(100, 100)):
    images = []
    labels = []
    for label, class_name in enumerate(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                image = io.imread(image_path)
                gray_image = color.rgb2gray(image)
                thresh = threshold_otsu(gray_image)
                binary_image = gray_image > thresh
                image_resized = resize(binary_image, target_size, mode='constant')
                images.append(image_resized)
                labels.append(label)


    return np.stack(images), np.array(labels)

def extract_features(images):
    return np.array([image.reshape(-1) for image in images])

custom_data_directory = 'DaneTreningowe'
image_data, target_labels = load_custom_images(custom_data_directory)


X_train, X_test, y_train, y_test = train_test_split(image_data, target_labels.ravel(), test_size=0.1, random_state=2)


X_train_flattened = extract_features(X_train)
X_test_flattened = extract_features(X_test)

scaler = preprocessing.MinMaxScaler().fit(X_train_flattened)
X_train_scaled = scaler.transform(X_train_flattened)
X_test_scaled = scaler.transform(X_test_flattened)


knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train_scaled, y_train)



model_filename = 'modelpojazdow.pkl'
joblib.dump(knn_model, model_filename)

if knn_model.n_neighbors <= len(X_test_scaled):
    y_pred = knn_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Dokładność modelu KNN: {accuracy * 100:.2f}%")
else:
    print("Liczba sąsiadów jest większa niż liczba próbek testowych.")
