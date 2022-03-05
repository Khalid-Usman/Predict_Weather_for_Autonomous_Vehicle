import numpy as np
from tensorflow import keras as K
from sklearn.metrics import confusion_matrix


class Evaluation:
    """Evaluation on un-seen data"""

    def __init__(self, model_path, test_generator, test_steps, categories):
        self.model_path = model_path
        self.test_generator = test_generator
        self.test_steps = test_steps
        self.categories = categories

    def print_confusion_mat(self, y_true, y_pred):
        for i in range(len(self.categories)):
            print(self.categories[i])
            print(confusion_matrix(y_true[i], y_pred[i]))

    def predict_accuracy(self):
        model = K.models.load_model(self.model_path)
        gt_list = [[] for _ in range(len(self.categories))]
        pred_list = [[] for _ in range(len(self.categories))]

        for test_batch in self.test_generator:
            gt = test_batch[1]
            pred = model.predict(test_batch[0])

            [sub.extend(gt[gt_list.index(sub)]) for sub in gt_list]
            [sub.extend(pred[pred_list.index(sub)]) for sub in pred_list]

        print(type(pred_list))
        print(type(pred_list[0]))
        gt_list = [np.array(i).argmax(axis=-1) for i in gt_list]
        pred_list = [np.array(i).argmax(axis=-1) for i in pred_list]
        self.print_confusion_mat(gt_list, pred_list)
