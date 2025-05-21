from django.shortcuts import render
from .forms import ImageUploadForm
from .models import UploadedImage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load your trained model (adjust path as needed)
MODEL_PATH = 'models/vgg_model.h5'  # Update this path
model = load_model(MODEL_PATH)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    
    # Map your class indices to actual class names
    class_names = {0: 'Fruit', 1: 'Packages', 2: 'Vegetables'}  # Update with your classes
    class_name = class_names.get(predicted_class, 'Unknown')
    
    return class_name, confidence

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()
            
            # Make prediction
            img_path = uploaded_image.image.path
            class_name, confidence = predict_image(img_path)
            
            # Update the model with prediction
            uploaded_image.prediction = class_name
            uploaded_image.confidence = confidence
            uploaded_image.save()
            
            return render(request, 'result.html', {
                'image': uploaded_image,
                'class_name': class_name,
                'confidence': confidence * 100
            })
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})