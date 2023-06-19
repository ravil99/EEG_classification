import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros


def fix_df(df: pd.DataFrame, zero_list_function, MAX_LEN: int):

    # transposing and setting correct index names
    df = df.T
    correct_indexes = [i+1 for i in range(len(df.columns))]
    df.columns = correct_indexes
    df[0] = df.index
    df = df.reset_index()

    # removing ['index'] column, setting the correct datatype and element order
    df = df.drop(['index'], axis=1)
    correct_indexes = df.columns.to_list()
    correct_indexes = correct_indexes[-1:] + correct_indexes[:-1]
    df = df[correct_indexes]

    # In case of errors='coerce', error-cell will
    # be turned into nan. Then it will be handled

    df[0] = pd.to_numeric(df[0], errors='coerce')
    df = df.fillna(df.mean())

    # ADD EXTRA ROWS HERE
    foundation = [zero_list_function(31) for i in range(MAX_LEN - len(df))]
    foundation = pd.DataFrame(foundation)
    df = df.append(foundation, ignore_index=True)

    return df


class CV_iteration():
    def __init__(self, name: str, train_set: list, valid_set: list, test_set: list):
        self.name = name
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set


class CustomDataset(Dataset):
    def __init__(self, filenames: list, classes: dict, MAX_LEN: int, is_fixed: bool):
        self.filenames = filenames
        self.classes = classes
        self.MAX_LEN = MAX_LEN
        self.is_fixed = is_fixed

    def __len__(self):
        return int(len(self.filenames))

    def __getitem__(self, idx):

        data = pd.read_csv(self.filenames[idx])

        if self.is_fixed == False:
            data = fix_df(data, zerolistmaker, self.MAX_LEN)
        # data = torch.from_numpy(np.asarray(data)).float()
        data = torch.from_numpy(np.moveaxis(np.asarray(data), 0, 1)).float()

        # OLD VERSION --> Converting DataFrame file to LIST OF TENSORS
        # data = [torch.tensor(df_data)
        #         for (df_name, df_data) in data.iteritems()]

        splitted_name = self.filenames[idx].split('/')[2].split('_')
        class_label = splitted_name[0]

        label = self.classes[class_label]
        if idx == self.__len__():
            raise IndexError

        return data, label
