from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Input, Flatten
from tensorflow.keras.models import Model


class Architecture:
    """"Create branching model"""

    def __init__(self, input_shape=None, filters=32, num_initial_blocks=3, num_branching_blocks=2, num_branches=7,
                 num_classes=None, categories=None):

        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = (3, 3)
        self.pool_size = (2, 2)
        self.padding = 'same'
        self.num_initial_blocks = num_initial_blocks
        self.num_branching_blocks = num_branching_blocks
        self.categories = categories
        if num_classes is None:
            self.num_classes = [3] * num_branches
        else:
            self.num_classes = num_classes

    def model_base(self, inputs):
        """Base model which is root for every branch"""
        for i in range(0, self.num_initial_blocks):
            if i == 0:
                x = Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding)(inputs)
            else:
                x = Conv2D(filters=self.filters * (i + 1), kernel_size=self.kernel_size, padding=self.padding)(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=self.pool_size)(x)
        return x

    def model_branch(self, inputs, num_classes, decider, model_name):
        """Branching models"""
        x = self.model_base(inputs)
        for i in range(0, self.num_branching_blocks):
            x = Conv2D(filters=self.filters * (i + 2 + self.num_branching_blocks), kernel_size=self.kernel_size,
                       padding=self.padding)(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=self.pool_size)(x)

        x = Flatten()(x)
        x = Dense(512)(x)
        x = Activation("relu")(x)
        x = Dense(num_classes)(x)
        x = Activation(decider, name=model_name)(x)
        return x

    def assemble_full_model(self):
        """Connecting all the branches with base model"""
        inputs = Input(shape=self.input_shape)
        outputs = list(map(lambda i: self.model_branch(inputs, self.num_classes[self.categories.index(i)],
                                                       decider='softmax',
                                                       model_name=i.lower()+"_output"),
                           self.categories))
        model = Model(inputs=inputs, outputs=outputs)
        branch_names = [i.lower() + "_output" for i in self.categories]
        loss_dict = dict(map(lambda x: (x, 'categorical_crossentropy') if self.num_classes[branch_names.index(x)] > 2
        else (x, 'binary_crossentropy'), branch_names))
        metric_dict = dict(map(lambda x: (x, 'accuracy'), branch_names))
        model.compile(optimizer='adam', loss=loss_dict, metrics=metric_dict)

        return model
