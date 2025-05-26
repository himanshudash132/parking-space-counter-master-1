import os           # OS module to interact with the file system
import pickle       # For saving and loading model objects

from skimage.io import imread           # To read images from disk
from skimage.transform import resize     # To resize image arrays for uniformity
import numpy as np                      # For numerical array operations
from sklearn.model_selection import train_test_split, GridSearchCV  # For dataset splitting and hyperparameter search
from sklearn.svm import SVC             # Support Vector Classifier algorithm
from sklearn.metrics import accuracy_score  # To evaluate model accuracy

# Directory containing subfolders 'empty' and 'not_empty' with training images
input_dir = '/home/phillip/Desktop/todays_tutorial/19_parking_car_counter/code/clf-data'
# Define class categories corresponding to folder names
categories = ['empty', 'not_empty']

# Initialize lists to hold flattened image data and corresponding labels
data = []
labels = []

# Loop through each category folder to read and process images
for category_idx, category in enumerate(categories):
    category_path = os.path.join(input_dir, category)
    # Enumerate each image file in the category directory
    for file_name in os.listdir(category_path):
        img_path = os.path.join(category_path, category, file_name)
        # Read image into NumPy array (grayscale or color)
        img = imread(img_path)
        # Resize to a consistent 15x15 pixels
        img_resized = resize(img, (15, 15))
        # Flatten 2D (or 3D) image into 1D feature vector and store
        data.append(img_resized.flatten())
        # Use folder index as numeric label (0 for 'empty', 1 for 'not_empty')
        labels.append(category_idx)

# Convert lists to NumPy arrays for model input
data = np.asarray(data)
labels = np.asarray(labels)

# Split dataset into training and test sets (80/20 split), stratifying by label distribution
x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels
)

# Initialize Support Vector Classifier
classifier = SVC()

# Define hyperparameter grid for gamma and regularization parameter C
parameters = [
    {
        'gamma': [0.01, 0.001, 0.0001],
        'C': [1, 10, 100, 1000]
    }
]

# Set up GridSearch to find the best hyperparameters via cross-validation
grid_search = GridSearchCV(
    estimator=classifier,
    param_grid=parameters,
    cv=5,                # 5-fold cross-validation (default)
    verbose=1            # Print progress messages
)

# Fit grid search on training data to identify best model
grid_search.fit(x_train, y_train)

# Retrieve the best estimator found during hyperparameter search
best_estimator = grid_search.best_estimator_

# Predict labels for the held-out test set
y_prediction = best_estimator.predict(x_test)

# Compute accuracy: proportion of correct predictions
score = accuracy_score(y_test, y_prediction)
print(f"{score * 100:.2f}% of samples were correctly classified")

# Save the trained model to disk for later use ('model.p')
with open('model.p', 'wb') as model_file:
    pickle.dump(best_estimator, model_file)  # Persist model object
