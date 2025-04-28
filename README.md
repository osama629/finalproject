# Eye Disease Classification Using Deep Learning

## Project Overview
This project aims to classify different types of **eye diseases** from medical images using deep learning models. The model is trained on a labeled dataset of eye images and evaluated based on performance metrics like accuracy, precision, recall, and F1-score.

## Dataset
- **Source:** [Kaggle - Eye Diseases Classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
- The dataset includes various labeled images of different eye conditions.
- Data is organized in folders, each representing a disease class.

## Project Structure
- **Data Loading:** 
  - Google Drive is mounted and KaggleHub is used to fetch the dataset.
  - Images and labels are extracted from the dataset directories.
  
- **Preprocessing:**
  - Data augmentation techniques are applied (rotation, zooming, flipping) using `ImageDataGenerator` to improve model generalization.

- **Model Architecture:**
  - A **Sequential CNN model** built with:
    - BatchNormalization
    - GlobalAveragePooling2D
    - Dense layers with L2 regularization
    - Dropout for regularization
  - **Optimizer:** Adamax
  - **Loss Function:** Categorical Crossentropy

- **Training:**
  - Model trained with early stopping and data augmentation to prevent overfitting.

- **Evaluation:**
  - Metrics used include confusion matrix, classification report (precision, recall, f1-score), and accuracy plots.

## Technologies Used
- Python
- TensorFlow / Keras
- Pandas
- Matplotlib, Seaborn (for visualization)
- Scikit-learn (for metrics)

## How to Run
1. Mount your Google Drive (if running on Colab).
2. Install and import necessary libraries (TensorFlow, KaggleHub, etc.).
3. Download the dataset using KaggleHub.
4. Run each cell sequentially to load data, preprocess, train, and evaluate the model.

## Results
- The model achieves good performance across different eye disease classes.
- Evaluation plots and metrics are generated to assess model effectiveness.

## Future Improvements
- Experiment with pre-trained models (e.g., ResNet, EfficientNet).
- Perform hyperparameter tuning to improve accuracy.
- Apply techniques like learning rate scheduling or different optimizers.
