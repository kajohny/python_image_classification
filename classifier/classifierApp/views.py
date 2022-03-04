from django.shortcuts import render
from .models import Prediction
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np

def index(request):
    if request.method == "POST":
        categories = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck"
        }

        model = load_model("results/cifar10-model-v2.h5")
        response = {}

        img = Prediction()
        img.image = request.FILES['image']
        img.save()

        predict_img1 = image.load_img("images/" + str(request.FILES['image']), target_size = (32, 32))
        predict_img = image.img_to_array(predict_img1) 
        predict_img = np.expand_dims(predict_img, axis = 0) 
        result = model.predict(predict_img) 
        for i in range(10):
            if(result[0][i] == 1):
                img.result = categories[i]
        img.save()

        response['result'] = img.result
        if response['result'] is None:
            response['result'] = "This image can't be classified"    
        return render(request, 'index.html', response)

    return render(request, 'index.html')
