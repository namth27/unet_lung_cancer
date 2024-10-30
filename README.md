# U-Net Lung Cancer Segmentation

## Overview
This project implements a U-Net model for lung cancer segmentation from medical images. The U-Net architecture is widely used in biomedical image segmentation due to its ability to capture context and localize effectively.

## Dataset
The dataset used for training and evaluation can be found at the following link:
[Download Dataset](https://data.mendeley.com/datasets/5rr22hgzwr/1)

## Model Training
The U-Net model was trained on the aforementioned dataset using Google Colab. After training, the model achieved promising results in segmenting lung cancer regions.

### Trained Model
You can download the trained model from the following link:
[Download Trained Model](https://drive.google.com/file/d/1-CFpI3Cuecq6C-5wMq1TNmk57QfAcSbb/view)

## Deployment
After training, the model was optimized and deployed using Streamlit to create a user-friendly interface for image upload and prediction.

### Running the Streamlit App
1. Clone the repository
2. Install Streamlit and other dependencies:
  pip install streamlit
3. Run the Streamlit app:
  streamlit run app_interface.py
4. Open your web browser and navigate to http://localhost:8501 to access the application.

Usage
Upload an image of a lung scan using the provided interface.
The model will predict the segmentation of lung cancer regions in the image and display the results.
Contributing
Contributions are welcome! Please feel free to submit a pull request.

License
This project is licensed under the MIT License.

Acknowledgments
The dataset is provided by Mendeley Data.
Special thanks to the creators of the U-Net architecture and Streamlit for facilitating model deployment.
