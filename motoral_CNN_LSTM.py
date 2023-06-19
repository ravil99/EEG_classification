"""
    Script for running training and eval.
pipeline of CNN_LSTM model.
"""

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.classification import F1Score
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from utils_motoric import form_data_tree, custom_CV
from utils_motoric import CV_iteration, CustomDataset, check_dataloader, evaluate_iteration
from utils_motoric import CNN_LSTM
from utils_motoric import train

# Fixing the random seed for REPRODUCIBILITY
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

# Extra ways of REPRODUCIBILITY:
# https://pytorch.org/docs/stable/notes/randomness.html

# REPRODUCIBILITY For CUDA:
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset = "BCI_C_IV"

if dataset == "Inno":
    rootdir = "Motoric NeuroScience Data"
    files_in_a_tree = 569
    MAX_LEN = 10095
    patients_tuple = tuple(range(1, 16))
    # Can be changed to imaginary
    classes_tuple = ("left_real", "right_real")
    classes = {"left_real": 0, "right_real": 1}
    channel_names = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5', 'FC1',
                     'FC2', 'FC6', 'FT10', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5',
                     'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1',
                     'Oz', 'O2']
    start = 5000
    end = 10000
    input_dim = 32
elif dataset == "BCI_C_IV":
    rootdir = "BCI_C_IV/Train_cut"
    files_in_a_tree = 1296
    MAX_LEN = 1875
    patients_tuple = tuple(range(1, 10))
    classes_tuple = ("left", "right")
    classes = {"left": 0, "right": 1}
    channel_names = ['EEG-Fz', 'EEG', 'EEG', 'EEG', 'EEG', 'EEG', 'EEG', 'EEG-C3', 'EEG',
                     'EEG-Cz', 'EEG', 'EEG-C4', 'EEG', 'EEG', 'EEG', 'EEG', 'EEG',
                     'EEG', 'EEG', 'EEG-Pz', 'EEG', 'EEG', 'EOG-left', 'EOG-central', 'EOG-right']
    start = 500
    end = 1500
    input_dim = 25

# Forming data tree
classes_data = form_data_tree(rootdir, classes_tuple, patients_tuple)


# Dividing to folds
folds_slice = custom_CV(classes_data, files_in_a_tree)

iteration = CV_iteration('original', train_set=folds_slice[0]+folds_slice[1] +
                         folds_slice[2], valid_set=folds_slice[3], test_set=folds_slice[3][len(folds_slice[3])//2:])

evaluate_iteration(iteration, patients_tuple)

use_only = 'C'
l_freq = 8
h_freq = 12
l_trans_bandwidth = 0.5
h_trans_bandwidth = 0.5

# MAKING DATALOADER FOR OUR PIPELINE
train_dl = DataLoader(CustomDataset(
    filenames=iteration.train_set, classes=classes, MAX_LEN=MAX_LEN, dataset=dataset,
    channel_names=channel_names, use_only=use_only, start=start,
    end=end, l_freq=l_freq, h_freq=h_freq, l_trans_bandwidth=l_trans_bandwidth,
    h_trans_bandwidth=h_trans_bandwidth), batch_size=2, shuffle=True)

val_dl = DataLoader(CustomDataset(
    filenames=iteration.valid_set, classes=classes, MAX_LEN=MAX_LEN, dataset=dataset,
    channel_names=channel_names, use_only=use_only, start=start,
    end=end, l_freq=l_freq, h_freq=h_freq, l_trans_bandwidth=0.5,
    h_trans_bandwidth=0.5), batch_size=2, shuffle=True)

seq_length = check_dataloader(train_dl)

# CHANGE THIS to CHANGE TRAINING PARAMETERS
# Model parameters
if use_only == 'C':
    input_dim = 3
else:
    input_dim = 25

hidden_dim = 256
layer_dim = 1       # Number of LSTM layers
output_dim = 2

# Good value - between 0.2 and 0.5
dropout_rate = 0
# Training parameters
lr = 0.005
n_epochs = 100
best_accuracy = 0
patience, trials = 15, 0

log_folder = f'runs/Correct_SCALED_{input_dim}_Channels_FILTERED_{l_freq}_{h_freq}_FRACTION_{start}->{end}_CNN_LSTM_(32-64-128-256)_{layer_dim}_Embedded_Dropout={dropout_rate}_LR={lr}_+Binary_LOSS'
print(f'Model parameters are : {log_folder.split("/")[-1]}')


def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


model = CNN_LSTM(input_dim, hidden_dim, layer_dim,
                 output_dim, dropout_rate)

model = model.cuda()

# Default loss function - CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
# Default optimizer - RMSprop
opt = torch.optim.RMSprop(model.parameters(), lr=lr)


model_path = "model_dir/Simple_LSTM_1_real_4_20_best_model.pt"
# Reading Pretrained model
# model, opt, start_epoch = load_ckp(model_path, model, opt)

# TRAINING
writer = SummaryWriter(log_folder)
writer.add_text('TEXT', 'Start model training', 0)

# VISUALIZING MODEL ARCHITECTURE

visualize = True
if visualize == True:
    dev = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    summary(CNN_LSTM(input_dim, hidden_dim, layer_dim, output_dim, dropout_rate),
            input_size=(5, 5000, input_dim), col_names=["kernel_size", "input_size", "output_size", "num_params"],
            verbose=2)

# tensorboard --logdir=runs

train(model=model, train_dl=train_dl, val_dl=val_dl, n_epochs=n_epochs,
      patience=patience, opt=opt, lr=lr, criterion=criterion,
      writer=writer, best_accuracy=best_accuracy, model_name='LSTM_CNN_First_trial',
      checkpoint_dir='checkpoint_dir', model_dir='model_dir')
