import cv2
from tensorflow import keras as K
import numpy as np
import matplotlib.pyplot as plt


class Inference:
    """"Inference on unseen Image"""

    def __init__(self, image_path, image_width=64, image_height=64, model_path=None, data_dict={}, categories=[]):
        self.image_path = image_path
        self.image_width = image_width
        self.image_height = image_height
        self.model_path = model_path
        self.data_dict = data_dict
        self.categories = categories
        self.labels = ""

    def show_test(self, img):
        plt.imshow(np.squeeze(img))
        plt.title(self.labels)
        plt.show()

    def predict_inference(self):
        model = K.models.load_model(self.model_path)
        img = cv2.resize(cv2.cvtColor(cv2.imread(self.image_path, 1), cv2.COLOR_BGR2RGB),
                         (self.image_width, self.image_height))
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        res = dict((v, k) for k, v in self.data_dict.items())
        for i in range(len(pred)):
            if np.argmax(pred[i]) != 0:
                self.labels += self.categories[i] + " is " + res[np.argmax(pred[i])] + " "
        self.show_test(img)
