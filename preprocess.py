import os
import numpy as np
import pandas as pd
from CustomImageGenerator import DataGenerator
from pandas.api.types import is_numeric_dtype as IND
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def create_data_frame(img_name, img_path, gt):
    df = pd.DataFrame({'img_name': img_name, 'img_path': img_path, 'Blockage': gt[:, 0], 'Blur': gt[:, 1],
                       'Fog': gt[:, 2], 'Impact_of_Sun': gt[:, 3], 'Precipitation': gt[:, 4], "Splashes": gt[:, 5],
                       "Sun_Ray": gt[:, 6]
                       })
    return df


def save_data(full_path, img_path, img_names, labels):
    gt = np.array(labels)
    df = create_data_frame(img_names, img_path, gt)
    # print(df.shape)
    # df = (df.groupby('img_name', as_index=False).agg(lambda x: x.sum() if IND(x) else ', '.join(x).split(',')[0]))
    df = df.groupby('img_name').agg({'img_path': 'first', 'Blockage': 'sum', 'Blur': 'sum', 'Fog': 'sum',
                                     'Impact_of_Sun': 'sum', 'Precipitation': 'sum', 'Splashes': 'sum', 'Sun_Ray': 'sum'
                                     })
    # print(df.shape)
    df.to_csv(full_path, sep=',')
    print('data written!')


def prepare_data(path, full_file_path, remove_dirs, dataset_dict, categories):
    img_path = []
    img_names = []
    labels = []

    main_dirs = os.listdir(path)
    for temp_dir in remove_dirs:
        main_dirs.remove(temp_dir)
    for main_dir in main_dirs:
        sub_dirs = os.path.join(path, main_dir)
        for sub_dir in os.listdir(sub_dirs):
            full_path = os.path.join(sub_dirs, sub_dir)
            if len(os.listdir(full_path)) > 0:
                for img_name in os.listdir(full_path):
                    img_names.append(img_name)
                    img_path.append(os.path.join(full_path, img_name))
                    label = [0] * len(categories)
                    label[categories.index(main_dir.replace(" ", "_"))] = dataset_dict[sub_dir]
                    labels.append(label)
    save_data(full_file_path, img_path, img_names, labels)


def get_stratified_data(X, y, test_size):
    X_train, X_test, y_train, y_test = [], [], [], []
    msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=0)
    for train_index, test_index in msss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test


def split_data(full_path, train_path, valid_path, test_path, choice='systematic'):
    """
    Split the full dataframe into training, validation and test

    choice
    ---------
    random: Select indices for train, valid and test data with percentage 70, 20, and 10 respectively.
    stratified: Use an external library which preprocesses data before to feed it to network.
    systematic: Select every n-th indices for train, valid and test data with percentage 70, 20, and 10 respectively.
    """
    df = pd.read_csv(full_path, index_col=False)
    if choice == 'random':
        probs = np.random.rand(len(df))
        training_mask, validation_mask, test_mask = probs < 0.7, (probs >= 0.7) and (probs < 0.9), probs >= 0.9
        df_train, df_valid, df_test = df[training_mask], df[validation_mask], df[test_mask]
    elif choice == 'stratified':
        X, y = df.iloc[:, :2].values, df.iloc[:, 2:].values
        X_train, X_valid, y_train, y_valid = get_stratified_data(X, y, test_size=0.2)
        df_valid = create_data_frame(X_valid[:, 0], X_valid[:, 1], y_valid)
        X_train, X_test, y_train, y_test = get_stratified_data(X_train, y_train, test_size=0.1)
        df_train = create_data_frame(X_train[:, 0], X_train[:, 1], y_train)
        df_test = create_data_frame(X_test[:, 0], X_test[:, 1], y_test)
    else:
        df_valid = df[(df.index % 10 == 8) | (df.index % 10 == 9)]
        df_test = df.iloc[1::10, :]
        df_train = df[(df.index % 10 != 0) & (df.index % 10 < 8)]

    # print(df_train.shape, df_valid.shape, df_test.shape)
    df_train.to_csv(train_path, sep=',', index=False)
    df_valid.to_csv(valid_path, sep=',', index=False)
    df_test.to_csv(test_path, sep=',', index=False)


def load_custom_data_gen(image_width=600, image_height=600, batch_size=32, num_branches=7, num_classes=None,
                         train_path=None, train_ratio=None, valid_path=None, valid_ratio=None):
    train_df = pd.read_csv(train_path)
    train_generator = DataGenerator(train_df,
                                    batch_size=batch_size,
                                    n_branches=num_branches,
                                    n_classes=num_classes,
                                    shuffle=True,
                                    image_width=image_width,
                                    image_height=image_height,
                                    ratios=train_ratio)

    valid_df = pd.read_csv(valid_path)
    valid_generator = DataGenerator(valid_df,
                                    batch_size=batch_size,
                                    n_branches=num_branches,
                                    n_classes=num_classes,
                                    shuffle=True,
                                    image_width=image_width,
                                    image_height=image_height,
                                    ratios=valid_ratio)

    return train_generator, valid_generator


def load_test_data_gen(image_width=600, image_height=600, batch_size=8, num_branches=7, num_classes=None,
                       test_path=None, train_ratio=None):
    test_df = pd.read_csv(test_path)
    test_generator = DataGenerator(test_df,
                                   batch_size=batch_size,
                                   n_branches=num_branches,
                                   n_classes=num_classes,
                                   shuffle=True,
                                   image_width=image_width,
                                   image_height=image_height,
                                   ratios=train_ratio)
    return test_generator
