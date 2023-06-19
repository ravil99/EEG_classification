import pandas as pd
import numpy as np
import mne
import torch
import pywt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler


def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros


def fix_df(df: pd.DataFrame, zero_list_function, dataset: str, MAX_LEN: int,
           channel_names: list, use_only: str, start: int, end: int,
           l_freq: float, h_freq: float, l_trans_bandwidth: float,
           h_trans_bandwidth: float):

    df = df.drop(['Unnamed: 0'], axis=1)
    if dataset == "Inno":
        numb_of_channels = 32
        sfreq = 1000
        main_channels = ['C3', 'Cz', 'C4']
    elif dataset == "BCI_C_IV":
        numb_of_channels = 25
        sfreq = 250
        main_channels = ['EEG-C3', 'EEG-Cz', 'EEG-C4']

    df.columns = list(range(numb_of_channels))
    # ADD EXTRA ROWS HERE
    foundation = [zero_list_function(numb_of_channels)
                  for i in range(MAX_LEN - len(df))]
    foundation = pd.DataFrame(foundation)
    df = pd.concat([df, foundation], ignore_index=True, axis=0)

    channel_types = ['eeg']*numb_of_channels
    ch_indices = [i for i, ch_name in enumerate(
        channel_names) if ch_name in main_channels]

    channel_names = [channel_names[i] for i in ch_indices]
    channel_types = [channel_types[i] for i in ch_indices]

    if use_only == 'C':
        # Drop some channels - 3
        np_eeg = df.iloc[:, ch_indices].to_numpy().T
    else:
        # Keep all channels - 32 or 25
        np_eeg = df.to_numpy().T

    # Baseline correction

    if dataset == "Inno":
        times = np.linspace(0, 10.094, 10095)
        np_eeg = mne.baseline.rescale(
            np_eeg, times, (0., 5.0), mode='mean', copy=False, verbose=False)

    # Let's just apply default FIR filtering
    np_eeg = mne.filter.filter_data(np_eeg, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
                                    l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth,
                                    copy=False, verbose=False)

    df = pd.DataFrame(np_eeg.T)
    # Only select desired range
    df = df[start:end]
    # Apply scaling
    scaler = RobustScaler()
    df = scaler.fit_transform(df)
    return df

def wavelet_fix_df(df: pd.DataFrame, zero_list_function, dataset: str, MAX_LEN: int,
           channel_names: list, use_only: str, start: int, end: int,
           l_freq: float, h_freq: float, l_trans_bandwidth: float,
           h_trans_bandwidth: float):

    df = df.drop(['Unnamed: 0'], axis=1)
    if dataset == "Inno":
        numb_of_channels = 32
        sfreq = 1000
        main_channels = ['C3', 'Cz', 'C4']
    elif dataset == "BCI_C_IV":
        numb_of_channels = 25
        sfreq = 250
        main_channels = ['EEG-C3', 'EEG-Cz', 'EEG-C4']

    df.columns = list(range(numb_of_channels))
    # ADD EXTRA ROWS HERE
    foundation = [zero_list_function(numb_of_channels)
                  for i in range(MAX_LEN - len(df))]
    foundation = pd.DataFrame(foundation)
    df = pd.concat([df, foundation], ignore_index=True, axis=0)

    channel_types = ['eeg']*numb_of_channels
    ch_indices = [i for i, ch_name in enumerate(
        channel_names) if ch_name in main_channels]

    channel_names = [channel_names[i] for i in ch_indices]
    channel_types = [channel_types[i] for i in ch_indices]

    if use_only == 'C':
        # Drop some channels - 3
        np_eeg = df.iloc[:, ch_indices].to_numpy().T
    else:
        # Keep all channels - 32 or 25
        np_eeg = df.to_numpy().T

    # # Baseline correction
    # if dataset == "Inno":
    #     times = np.linspace(0, 10.094, 10095)
    #     np_eeg = mne.baseline.rescale(
    #         np_eeg, times, (0., 5.0), mode='mean', copy=False, verbose=False)

    # Let's just apply default FIR filtering
    np_eeg = mne.filter.filter_data(np_eeg, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq,
                                    l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth,
                                    copy=False, verbose=False)

    df = pd.DataFrame(np_eeg.T)
    # Only select desired range
    df = df[start:end]

    # # Apply scaling
    # scaler = RobustScaler()
    # df = scaler.fit_transform(df)

    return df


class CV_iteration():
    def __init__(self, name: str, train_set: list, valid_set: list, test_set: list):
        self.name = name
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set


def evaluate_iteration(iteration: CV_iteration, patients_tuple: tuple):
    def check_distribution(part_of_dataset):
        patients_dict = {}
        for i in patients_tuple:
            patients_dict[i] = 0

        for i in patients_dict:
            for j in part_of_dataset:
                # Getting the patient №
                if int(j.split('/')[-1].split('_')[0]) == i:
                    patients_dict[i] = patients_dict[i] + 1
        return patients_dict
    print(f'''Distribution of patients among train set of {iteration.name} is \n {check_distribution(iteration.train_set)} \n
              Distribution of patients among valid set of {iteration.name} is \n {check_distribution(iteration.valid_set)} \n
              Distribution of patients among test set of {iteration.name} is \n {check_distribution(iteration.test_set)} \n ''')


class CustomDataset(Dataset):
    def __init__(self, filenames: list, classes: dict, dataset: str,
                 MAX_LEN: int, channel_names: list, use_only: str,
                 start: int, end: int, l_freq: float, h_freq: float,
                 l_trans_bandwidth: float, h_trans_bandwidth: float, model: str = 'LSTM'):

        self.filenames = filenames
        self.classes = classes
        self.dataset = dataset
        self.MAX_LEN = MAX_LEN
        self.channel_names = channel_names
        self.use_only = use_only
        self.start = start
        self.end = end
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.model = model

    def __len__(self):
        return int(len(self.filenames))

    def __getitem__(self, idx):

        data = pd.read_csv(self.filenames[idx])
        data = fix_df(data, zerolistmaker, self.dataset, self.MAX_LEN,
                      self.channel_names, self.use_only,
                      self.start, self.end, self.l_freq,
                      self.h_freq, self.l_trans_bandwidth,
                      self.h_trans_bandwidth)

        if self.model == 'LSTM':
            data = torch.from_numpy(np.asarray(data)).float()
        elif self.model == 'ChronoNet':
            data = torch.from_numpy(np.moveaxis(
                np.asarray(data), 0, 1)).float()

        splitted_name = self.filenames[idx].split('/')
        if self.dataset == "Inno":
            class_label = splitted_name[1]
        elif self.dataset == "BCI_C_IV":
            class_label = splitted_name[2]

        label = self.classes[class_label]
        if idx == self.__len__():
            raise IndexError

        return data, label
    
class WaveletCustomDataset(Dataset):
    def __init__(self, filenames: list, classes: dict, dataset: str,
                 MAX_LEN: int, channel_names: list, use_only: str,
                 start: int, end: int, l_freq: float, h_freq: float,
                 l_trans_bandwidth: float, h_trans_bandwidth: float, model: str = 'LSTM'):

        self.filenames = filenames
        self.classes = classes
        self.dataset = dataset
        self.MAX_LEN = MAX_LEN
        self.channel_names = channel_names
        self.use_only = use_only
        self.start = start
        self.end = end
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.l_trans_bandwidth = l_trans_bandwidth
        self.h_trans_bandwidth = h_trans_bandwidth
        self.model = model

    def __len__(self):
        return int(len(self.filenames))

    def __getitem__(self, idx):

        data = pd.read_csv(self.filenames[idx])
        data = wavelet_fix_df(data, zerolistmaker, self.dataset, self.MAX_LEN,
                      self.channel_names, self.use_only,
                      self.start, self.end, self.l_freq,
                      self.h_freq, self.l_trans_bandwidth,
                      self.h_trans_bandwidth)

        # Wavelet transform generates time-frequency graph

        sampling_freq = 250

        wavlist = pywt.wavelist(kind='continuous')
        print("Class of continuous wavelet functions：")
        print(wavlist)
        t = np.arange(2,6.5,1.0/sampling_freq)
        wavename = 'morl'    # "cmorB-C" where B is the bandwidth and C is the center frequency.
        # frequencies = pywt.scale2frequency('cmor1.5-0.5', [1, 2, 3, 4]) / (1/sampling_rate)
        # print(frequencies)
        totalscal = 64    # scale 
        fc = pywt.central_frequency(wavename) #  central frequency
        cparam = 2 * fc * totalscal
        scales = cparam/np.arange(1,totalscal+1)

        # Generate image here

        [cwtmatr3, frequencies3] = pywt.cwt(data[0],scales,wavename,1.0/sampling_freq) 
        [cwtmatr4, frequencies4] = pywt.cwt(data[2],scales,wavename,1.0/sampling_freq) 

        cwtmatr = np.concatenate([abs(cwtmatr3[7:30,:]), abs(cwtmatr4[7:30,:])],axis=0)

        if self.model == 'LSTM':
            data = torch.from_numpy(np.asarray(data)).float()
        elif self.model == 'ChronoNet':
            data = torch.from_numpy(np.moveaxis(
                np.asarray(data), 0, 1)).float()

        splitted_name = self.filenames[idx].split('/')
        if self.dataset == "Inno":
            class_label = splitted_name[1]
        elif self.dataset == "BCI_C_IV":
            class_label = splitted_name[2]

        label = self.classes[class_label]
        if idx == self.__len__():
            raise IndexError

        return data, label


def check_dataloader(train_dl):
    for i, (data, label) in enumerate(train_dl):
        print(
            f"Our DataLoader returns {type(data)} and it's example is {data.shape}")
        seq_length = data.shape[2]
        print(f'Seq_lenght is {seq_length}')
        print(f'Type of data is {data[1,:,-1][-1].dtype}')
        print(f"Our label is {label[-1]} and it's type is {label[-1].dtype}")
        if i == 0:
            return seq_length
