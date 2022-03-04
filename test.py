from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np

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

test_img1 = image.load_img('images/airplane1.png', target_size = (32, 32))
test_img = image.img_to_array(test_img1) 
test_img = np.expand_dims(test_img, axis = 0) 
result = model.predict(test_img) 
for i in range(10):
    if(result[0][i] == 1):
        print(categories[i])