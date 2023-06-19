import os


def form_data_tree(rootdir: str, classes_tuple: tuple, patients_tuple: tuple):
    # This is the main object with data
    classes_data = {}                       # Dict with lists

    for dirs, subdirs, files in os.walk(rootdir):
        key = dirs.split('/')[-1]
        if key in classes_tuple:
            patients_dict = {}
            for i in patients_tuple:
                patients_dict[i] = []
            classes_data[key] = patients_dict
            for file in files:
                patient_number = int(file.split('_')[1])
                data_link = os.path.join(dirs, file)
                classes_data[key][patient_number].append(data_link)
    return classes_data


def custom_CV(classes_data: dict):
    number_of_folds = 5
    average_length_of_list = int(4000 / (20*8))         # 25

    train_set = []
    valid_set = []
    test_set = []

    fold_1, fold_2, fold_3, fold_4, fold_5 = [], [], [], [], []

    for i in classes_data.values():
        for j in i.values():
            for filename in j[:number_of_folds]:
                fold_1.append(filename)
            for filename in j[number_of_folds: 2 * number_of_folds]:
                fold_2.append(filename)
            for filename in j[2 * number_of_folds: 3 * number_of_folds]:
                fold_3.append(filename)
            for filename in j[3 * number_of_folds: 4 * number_of_folds]:
                fold_4.append(filename)
            for filename in j[4 * number_of_folds:]:
                fold_5.append(filename)

    return fold_1, fold_2, fold_3, fold_4, fold_5
