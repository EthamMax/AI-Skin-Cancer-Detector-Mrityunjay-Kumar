# AI-Powered Skin Cancer Detection Web App

**Developed by:** Mrityunjay Kumar  
**College:** Acharya Narendra Dev College, University of Delhi

## Project Overview
This project is an interactive web app that uses a deep learning model to analyze skin lesion images and predict the likelihood of melanoma (skin cancer). It also includes a Grad-CAM visualization to show which regions of the image the model focused on.

## Features
- Upload skin lesion images
- Predict the type of skin lesion
- Visualize Grad-CAM heatmaps for explainability

## Technologies Used
- TensorFlow / Keras
- Streamlit
- OpenCV & Pillow
- Google Colab for training
- Streamlit Sharing / Hugging Face Spaces for deployment

## How to Run
1. Clone the repository.
2. Install the required packages with `pip install -r requirements.txt`.
3. Run the app with `streamlit run app.py`.

## License
This project is open source under the MIT License.

## Acknowledgements
Special thanks to [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) for providing the HAM10000 dataset.
