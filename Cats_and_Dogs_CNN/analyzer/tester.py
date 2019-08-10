from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
from IPython.display import display
from PIL import Image

# load json and create model
json_file = open('../models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("../models/model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

test_image = image.load_img('../test/20180209_093454.jpg', target_size=(64, 64))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = loaded_model.predict(test_image)
if result[0][0] > 0.5:
    print('dog')
else:
    print('cat')