import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import F1Score
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from utils import form_data_tree, custom_CV, fix_df, zerolistmaker
from utils import CV_iteration, CustomDataset

rootdir = "Scaled NeuroScience Data split/data split 2"

classes_tuple = ("a15", "a25", "a40", "a45", "a55", "a60", "a75", "a85")
patients_tuple = tuple(range(1, 21))

MAX_LEN = 1500
classes = {"a15": 0, "a25": 0, "a40": 1, "a45": 1,
           "a55": 1, "a60": 1, "a75": 0, "a85": 0}

# Forming data tree
classes_data = form_data_tree(rootdir, classes_tuple, patients_tuple)

# Dividing to folds
folds_slice = custom_CV(classes_data)

iteration = CV_iteration('original', train_set=folds_slice[0]+folds_slice[1] +
                         folds_slice[2], valid_set=folds_slice[3], test_set=folds_slice[4])


# MAKING DATALOADER FOR OUR PIPELINE
train_dl = DataLoader(CustomDataset(
    filenames=iteration.train_set, classes=classes, MAX_LEN=MAX_LEN, is_fixed=True), batch_size=20, shuffle=True)
val_dl = DataLoader(CustomDataset(
    filenames=iteration.valid_set, classes=classes, MAX_LEN=MAX_LEN, is_fixed=True), batch_size=20, shuffle=True)

print(iteration.train_set)

# Checking dataloader
for i, (data, label) in enumerate(train_dl):
    print(
        f"Our DataLoader returns {type(data)} and it's example is {data.shape}")
    seq_length = data.shape[2]
    print(f'Seq_lenght is {seq_length}')
    print(f'Our DataLoader looks like {data}')
    print(f"Our label is {label} and it's type is {type(label)}")
    if i == 0:
        break

# Model parameters
first_gru_length = seq_length // 8
output_dim = 3

# Training parameters
lr = 0.05
n_epochs = 50
best_accuracy = 0
patience, trials = 15, 0

# Change it
log_folder = 'runs/Chrononet_Scaled_split2_2_classes'

class CNBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(CNBlock, self).__init__()

        self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding

        self.conv1 = nn.Conv1d(in_channels= self.in_channels, out_channels= 32,kernel_size= 2, stride= 2,padding= 0)
        self.conv2 = nn.Conv1d(in_channels= self.in_channels, out_channels= 32,kernel_size= 4, stride= 2,padding= 1)
        self.conv3 = nn.Conv1d(in_channels= self.in_channels, out_channels= 32,kernel_size= 8, stride= 2,padding= 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1,x2,x3], dim=1)
        return x

class ChronoNet(nn.Module):
    def __init__(self, first_gru_length, out_dim = 8):
        super().__init__()

        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding

        self.block1 = CNBlock(31)
        self.block2 = CNBlock(96)
        self.block3 = CNBlock(96)

        self.gru1 = nn.GRU(input_size= 96, hidden_size=32, batch_first = True)
        self.gru2 = nn.GRU(input_size= 32, hidden_size=32, batch_first = True)
        self.gru3 = nn.GRU(input_size= 64, hidden_size=32, batch_first = True)
        self.gru4 = nn.GRU(input_size= 96, hidden_size=32, batch_first = True)
        self.gru_linear = nn.Linear(first_gru_length,1)
        self.flatten = nn.Flatten()
        self.fcl = nn.Linear(32,out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.permute(0,2,1)

        gru_out1,_ = self.gru1(x)
        gru_out2,_ = self.gru2(gru_out1)
        gru_out = torch.cat([gru_out1,gru_out2], dim=2)
        gru_out3,_ =self.gru3(gru_out)
        gru_out = torch.cat([gru_out1,gru_out2,gru_out3],dim=2)
        
        linear_out=self.relu(self.gru_linear(gru_out.permute(0,2,1)))
        gru_out4,_=self.gru4(linear_out.permute(0,2,1))
        x=self.flatten(gru_out4)
        x = self.fcl(x)
        return x

model = ChronoNet(first_gru_length=first_gru_length, out_dim=output_dim)
model = model.cuda()
# nn.CrossEntropyLoss is a loss function, that applies Softmax automatically
criterion = nn.CrossEntropyLoss()
opt = torch.optim.RMSprop(model.parameters(), lr=lr)

# VISUALIZING MODEL ARCHITECTURE
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# TRAINING

writer = SummaryWriter(log_folder)
writer.add_text('TEXT', 'Start model training', 0)

# tensorboard --logdir=runs --> For tensorboard

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
    for x_val, y_val in val_dl:
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
        torch.save(model.state_dict(), f'{str(lr)}_best.pth')
        print(f'Epoch {epoch} best model saved with Accuracy: {best_accuracy}')
    else:
        trials += 1
        if trials >= patience:
            print(f'Early stopping on epoch {epoch}')
            break
print('The training is finished! Restoring the best model weights')
writer.flush()

# 