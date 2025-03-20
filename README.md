# Sign Language Detector  

A Python-based Sign Language Detector for recognizing hand gestures. This project involves collecting images, creating a dataset, training a model, and running inference to detect sign language gestures.  

## Requirements  

Ensure you have the following installed:  

- Python 3.8+  
- OpenCV  
- TensorFlow/Keras  
- NumPy, Pandas, and other required dependencies (install using `requirements.txt`)  

## Installation  

### Installation Steps  

1. **Clone this repository**  
   ```
   git clone https://github.com/your-username/sign-language-detector.git
   ```
   
2. **Enter the project directory**
   ```
   cd sign-language-detector
   ```
   
4. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
   
5. **Run the image collection script**
   ```
   python collect_imgs.py
   ```
   
   - This will collect images and save them in the ```/data``` folder.
   
7. **Create the dataset**
   ```
   python create_dataset.py
   ```

   - This generates ```data.pickle``` for model training.
     
9. **Train the classifier**
    ```
    python train_classifier.py
    ```

    - This step will train the model using the prepared dataset.
      
11. **Test the model**
    ```
    python inference_classifier.py
    ```

    - This runs the trained classifier for sign language detection.

    ## Usage

    Once trained, the model can recognize hand gestures based on the collected images.

    ## Further Details
    For a detailed tutorial, refer to this YouTube video:
    [Sign Language Detection Tutorial](https://www.youtube.com/watch?v=MJCSjXepaAM "Watch the full tutorial on YouTube")
