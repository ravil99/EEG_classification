"""
    Script for running training and eval.
pipeline of Chrononet model.
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
from utils_motoric import ChronoNet
from utils_motoric import train

# Fixing the random seed for REPRODUCIBILITY
torch.manual_seed(1)
np.random.seed(1)

rootdir = "Motoric NeuroScience Data"

classes_tuple = ("left_real", "right_real")
patients_tuple = tuple(range(1, 16))

MAX_LEN = 10095
classes = {"left_real": 0, "right_real": 1}

# Forming data tree
classes_data = form_data_tree(rootdir, classes_tuple, patients_tuple)
files_in_a_tree = 569

# Dividing to folds
folds_slice = custom_CV(classes_data, files_in_a_tree)

iteration = CV_iteration('original', train_set=folds_slice[0]+folds_slice[1] +
                         folds_slice[2], valid_set=folds_slice[3], test_set=folds_slice[3][len(folds_slice[3])//2:])

# evaluate_iteration(iteration, patients_tuple)

channel_names = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT9', 'FC5', 'FC1',
                 'FC2', 'FC6', 'FT10', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5',
                 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1',
                 'Oz', 'O2']
use_only = 'C'

start = 5000
end = 10000
l_freq = 9
h_freq = 11
l_trans_bandwidth = 0.5
h_trans_bandwidth = 0.5

# MAKING DATALOADER FOR OUR PIPELINE
train_dl = DataLoader(CustomDataset(
    filenames=iteration.train_set, classes=classes, MAX_LEN=MAX_LEN,
    channel_names=channel_names, use_only=use_only, start=start,
    end=end, l_freq=l_freq, h_freq=h_freq, l_trans_bandwidth=l_trans_bandwidth,
    h_trans_bandwidth=h_trans_bandwidth, model='ChronoNet'), batch_size=2, shuffle=True)

val_dl = DataLoader(CustomDataset(
    filenames=iteration.valid_set, classes=classes, MAX_LEN=MAX_LEN,
    channel_names=channel_names, use_only=use_only, start=start,
    end=end, l_freq=l_freq, h_freq=h_freq, l_trans_bandwidth=l_trans_bandwidth,
    h_trans_bandwidth=h_trans_bandwidth, model='ChronoNet'), batch_size=2, shuffle=True)

seq_length = check_dataloader(train_dl)

# CHANGE THIS to CHANGE TRAINING PARAMETERS
# Model parameters
input_dim = 32
first_gru_length = seq_length // 8
output_dim = 2

# Training parameters
lr = 0.05 * 0.1
n_epochs = 50
best_accuracy = 0
patience, trials = 15, 0

log_folder = f'runs/SCALED_{input_dim}_Channels_FILTERED_{l_freq}_{h_freq}_FRACTION_ChronoNet_LR={lr}'
print(f'Model parameters are : {log_folder.split("/")[-1]}')

model = ChronoNet(first_gru_length=first_gru_length, input_dim=input_dim,
                  out_dim=output_dim)
model = model.cuda()

# Default loss function - CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
# Default optimizer - RMSprop
opt = torch.optim.RMSprop(model.parameters(), lr=lr)

# TRAINING
writer = SummaryWriter(log_folder)
writer.add_text('TEXT', 'Start model training', 0)

# VISUALIZING MODEL ARCHITECTURE
visualize = True
if visualize == True:
    dev = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    summary(ChronoNet(first_gru_length, input_dim, output_dim),
            input_size=(5, 1438, input_dim), col_names=["kernel_size", "input_size", "output_size", "num_params"],
            verbose=2)

# '''tensorboard --logdir=runs''' --> For tensorboard

train(model=model, train_dl=train_dl, val_dl=val_dl, n_epochs=n_epochs,
      patience=patience, opt=opt, lr=lr, criterion=criterion,
      writer=writer, best_accuracy=best_accuracy)
