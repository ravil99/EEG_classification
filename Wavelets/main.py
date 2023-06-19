import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torchmetrics.classification import F1Score
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from datasets import CustomDataset
from models import Net


import os
import pandas as pd

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

train_folder = 'images_train/'
test_folder = 'images_test/'

sub_folders = ["left", "right"]  # Insert your classes here
labels = [0, 1]

data_train = []
for s, l in zip(sub_folders, labels):
    for r, d, f in os.walk(train_folder + s):
        for file in f:
            if ".png" in file:
                data_train.append((os.path.join(s, file), l))

df_train = pd.DataFrame(data_train, columns=['file_name', 'label'])
print(df_train.shape)

data_test = []
for s, l in zip(sub_folders, labels):
    for r, d, f in os.walk(test_folder + s):
        for file in f:
            if ".png" in file:
                data_test.append((os.path.join(s, file), l))

df_test = pd.DataFrame(data_test, columns=['file_name', 'label'])
print(df_test.shape)

# Creating dataloaders
train_dl = CustomDataset(
    root_dir=train_folder, image_size=128, df=df_train)
test_dl = CustomDataset(
    root_dir=test_folder, image_size=128, df=df_test)

train_dl = DataLoader(dataset=train_dl, batch_size=10, shuffle=True,
                      num_workers=0)
test_dl = DataLoader(dataset=test_dl, batch_size=10, shuffle=True,
                     num_workers=0)

for (image_batch, label_batch) in train_dl:
    print(image_batch.shape)
    print(label_batch.shape)
    break


# Create an instance of the PyTorch model
num_classes = 2  # Replace with the appropriate number of classes
model = Net(num_classes)

# Print the model summary
print(model)

visualize = True
if visualize == True:
    dev = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    summary(Net(num_classes),
            input_size=(10, 3, 128, 128), col_names=["kernel_size", "input_size", "output_size", "num_params"],
            verbose=2)

model = model.cuda()

lr = 0.0001
n_epochs = 100
best_accuracy = 0
patience, trials = 15, 0

criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.0003)

log_folder = f'runs/Bigger_model_Spectrogramm_run_lr={lr}'

writer = SummaryWriter(log_folder)
writer.add_text('TEXT', 'Start model training', 0)

print('Start model training')
for epoch in range(1, n_epochs + 1):
    for i, (x_batch, y_batch) in enumerate(train_dl):
        model.train()
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        opt.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()
        opt.step()
    model.eval()
    correct, total = 0, 0
    for x_val, y_val in test_dl:
        x_val, y_val = [t.cuda() for t in (x_val, y_val)]
        # forward pass
        out = model(x_val)
        # softmax for classification
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_val.size(0)
        correct += (preds == y_val).sum().item()
    # f1 = F1Score(task="multiclass", num_classes=8)
    accuracy = correct/total
    writer.add_scalar("Accuracy_score", accuracy, epoch)
    if epoch % 5 == 0:
        print(
            f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Accuracy.: {accuracy}')
    if accuracy > best_accuracy:
        trials = 0
        best_accuracy = accuracy
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': opt.state_dict()
        }
        is_best = True
        # torch.save(model.state_dict(), f'{model_name}_best.pth')
        print(
            f'Epoch {epoch} best model saved with Accuracy: {best_accuracy}')
    else:
        trials += 1
        is_best = False
    if trials >= patience:
        print(f'Early stopping on epoch {epoch}')
        break

    print('The training is finished! Restoring the best model weights')
    writer.flush()

# tensorboard --logdir=runs
