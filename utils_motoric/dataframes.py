import os
from math import ceil


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
                patient_number = int(file.split('_')[0])
                data_link = os.path.join(dirs, file)
                classes_data[key][patient_number].append(data_link)
    return classes_data


def custom_CV(folder_tree: dict, files_in_a_tree: int):

    number_of_folds = 4
    # 9 for BCI...
    # 15 for Inno
    elements_per_list = files_in_a_tree/(9*2)
    part = ceil(elements_per_list/number_of_folds)

    fold_1, fold_2, fold_3, fold_4 = [], [], [], []

    # Classes_data - original Dataset:
    # It consists of 8*20 list of len = 25

    for i in folder_tree.values():
        for k, j in i.items():
            for filename in j[:part]:
                fold_1.append(filename)
            for filename in j[part: 2 * part]:
                fold_2.append(filename)
            for filename in j[2 * part: 3 * part]:
                fold_3.append(filename)
            for filename in j[3 * part:]:
                fold_4.append(filename)

    return (fold_1, fold_2, fold_3, fold_4)
