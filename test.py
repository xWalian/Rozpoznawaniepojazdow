import os

import joblib
from skimage import io, color
from skimage.filters import threshold_otsu
from skimage.transform import resize
from sklearn.metrics import accuracy_score


def check_images_in_directory(model, directory):
    true_labels = []
    predicted_labels = []

    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            true_label = class_name.lower()
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                test_image = io.imread(image_path)
                gray_image = color.rgb2gray(test_image)
                thresh = threshold_otsu(gray_image)
                binary_image = gray_image > thresh
                test_image_resized = resize(binary_image, (100, 100), mode='constant')
                test_image_flattened = test_image_resized.flatten()
                predicted_label = model.predict([test_image_flattened])[0]
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)

                if model.n_neighbors <= len(test_image_flattened):
                    if predicted_label == 0:
                        print(f"{image_path}: Na obrazie jest samochód.")
                    elif predicted_label == 1:
                        print(f"{image_path}: Na obrazie jest motocykl.")


    true_labels = [0 if label == 'cars' else 1 for label in true_labels]

    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Skuteczność wyszukiwania: {accuracy * 100:.2f}%")


directory_to_check = 'Danedotestowania'
loaded_model = joblib.load('modelpojazdow.pkl')
check_images_in_directory(loaded_model, directory_to_check)
