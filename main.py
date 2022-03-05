from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from preprocess import prepare_data, split_data, load_custom_data_gen, load_test_data_gen
from CustomModel import Architecture
import json
import os
from predict import Evaluation
from inference import Inference


def train_model():
    """Load JSON parameters"""
    with open('config.JSON') as file:
        data = json.load(file)

    print("Preparing data...")
    prepare_data(data['path'], data['full_path'], data['remove_dirs'], data['dataset_dict'], data['categories'])

    print("Splitting data...")
    split_data(data['full_path'], data['train_path'], data['valid_path'], data['test_path'], data['split_data'])

    print('Loading data...')
    train_gen, valid_gen = load_custom_data_gen(data['image_width'], data['image_height'], data['batch_size'],
                                                data['num_branches'], data['num_classes'],
                                                train_path=data['train_path'], train_ratio=data['sampling_train_ratio'],
                                                valid_path=data['valid_path'], valid_ratio=data['sampling_valid_ratio'])

    print('Building model ...')
    input_shape = (data['image_height'], data['image_width'], data['image_channel'])
    model = Architecture(input_shape=input_shape, filters=data['filters'],
                         num_initial_blocks=data['num_initial_blocks'],
                         num_branching_blocks=data['num_branching_blocks'],
                         num_branches=data['num_branches'],
                         num_classes=data['num_classes'], categories=data['categories']).assemble_full_model()
    print(model.summary())

    print('Training model')
    step_size_train = train_gen.n // train_gen.batch_size
    step_size_valid = valid_gen.n // valid_gen.batch_size
    tb = TensorBoard(log_dir=os.path.join('', 'logs'), update_freq=10)
    cp = ModelCheckpoint(data['model_checkpoint_path'], monitor='val_loss', verbose=1, save_best_only=True,
                         save_weights_only=False)
    es = EarlyStopping(monitor="val_loss", mode='min', verbose=1)
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=step_size_train,
                        validation_data=valid_gen,
                        validation_steps=step_size_valid,
                        epochs=data['epochs'],
                        verbose=1,
                        shuffle=True,
                        callbacks=[cp, es, tb])
    model.save(data['model_output_path'])

    print("Evaluating model")
    test_gen = load_test_data_gen(data['image_width'], data['image_height'], data['batch_size'], data['num_branches'],
                                  data['num_classes'], test_path=data['test_path'])
    step_size_test = test_gen.n // test_gen.batch_size
    eval = Evaluation(model_path=data['model_output_path'], test_generator=test_gen, test_steps=step_size_test,
                      categories=data['categories'])
    eval.predict_accuracy()

    print("Inference model")
    inf = Inference(image_path=data['test_image_path'], image_width=data['image_width'],
                    image_height=data['image_height'], model_path=data['model_output_path'],
                    data_dict=data['dataset_dict'], categories=data['categories'])
    inf.predict_inference()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_model()
