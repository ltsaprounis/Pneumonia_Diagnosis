import numpy as np
import os
import shutil


def train_val_split(val_split):
    """
    Function that creates a new directory with more validation images
    Need to generalise this so that it works for different problems (class names, more than 2 classes etc.)
    :param val_split: number from zero to 1
    :return:val_v2 directory
    """
    # replace data path with input from yml file!
    data_path = r"C:\Users\leoni\PycharmProjects\Data\Post_ETL"
    PATH_OLD = os.path.join(data_path, "chest_xray")
    PATH = os.path.join(data_path, "chest_xray_V2")
    # train, validate, test dirs
    train_dir = os.path.join(PATH, "train")
    val_dir = os.path.join(PATH, "val")
    val_new_dir = os.path.join(PATH, "val_v2")
    test_dir = os.path.join(PATH, "test")

    # train
    pneum_train_dir = os.path.join(train_dir, 'PNEUMONIA')
    norm_train_dir = os.path.join(train_dir, 'NORMAL')

    # val (move data into train dir and use validation splits)
    pneum_val_dir = os.path.join(val_dir, 'PNEUMONIA')
    norm_val_dir = os.path.join(val_dir, 'NORMAL')

    # val (move data into train dir and use validation splits)
    pneum_val_new_dir = os.path.join(val_new_dir, 'PNEUMONIA')
    norm_val_new_dir = os.path.join(val_new_dir, 'NORMAL')

    # test
    pneum_test_dir = os.path.join(test_dir, 'PNEUMONIA')
    norm_test_dir = os.path.join(test_dir, 'NORMAL')

    data_dict = {'pneum_train': len(os.listdir(pneum_train_dir)),
                 'norm_train': len(os.listdir(norm_train_dir)),
                 'pneum_val': len(os.listdir(pneum_val_dir)),
                 'norm_val': len(os.listdir(norm_val_dir)),
                 'pneum_test': len(os.listdir(pneum_test_dir)),
                 'norm_test': len(os.listdir(norm_test_dir))}

    # remove v2 directory if it's already there
    if os.path.exists(PATH) and os.path.isdir(PATH):
        shutil.rmtree(PATH)
    # copy directory to create the post processing dir
    shutil.copytree(PATH_OLD, PATH)

    # make the val new dir and class dirs
    os.mkdir(val_new_dir)
    os.mkdir(pneum_val_new_dir)
    os.mkdir(norm_val_new_dir)

    # move data from the train split to the val split
    for i in range(round(data_dict['pneum_train']*val_split)):
        dir_list = os.listdir(pneum_train_dir)
        file_idx = np.random.randint(low=0, high=len(dir_list))
        file_path = os.path.join(pneum_train_dir, dir_list[file_idx])
        # move file to new dir
        dest_path = os.path.join(pneum_val_new_dir, dir_list[file_idx])
        shutil.move(file_path, dest_path)

    # move data from the train split to the val split
    for i in range(round(data_dict['norm_train']*val_split)):
        dir_list = os.listdir(norm_train_dir)
        file_idx = np.random.randint(low=0, high=len(dir_list))
        file_path = os.path.join(norm_train_dir, dir_list[file_idx])
        # move file to new dir
        dest_path = os.path.join(norm_val_new_dir, dir_list[file_idx])
        shutil.move(file_path, dest_path)

    print('new validation directory with %0.2f of training data in %s' % (val_split, val_new_dir))


if __name__ == '__main__':
    train_val_split(0.1)
