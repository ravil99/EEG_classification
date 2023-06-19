import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import F1Score
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from utils import form_data_tree, custom_CV, fix_df, zerolistmaker
from utils import CV_iteration, CustomDataset

rootdir = "Scaled NeuroScience Data"


classes_tuple = ("a15", "a25", "a40", "a45", "a55", "a60", "a75", "a85")
patients_tuple = tuple(range(1, 21))

MAX_LEN = 1500
classes = {"a15": 0, "a25": 1, "a40": 2, "a45": 3,
           "a55": 4, "a60": 5, "a75": 6, "a85": 7}

# Forming data tree
classes_data = form_data_tree(rootdir, classes_tuple, patients_tuple)

# Dividing to folds
folds_slice = custom_CV(classes_data)

iteration = CV_iteration('original', train_set=folds_slice[0]+folds_slice[1] +
                         folds_slice[2], valid_set=folds_slice[3], test_set=folds_slice[4])


class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]


# MAKING DATALOADER FOR OUR PIPELINE
train_dl = DataLoader(CustomDataset(
    filenames=iteration.train_set, classes=classes, MAX_LEN=MAX_LEN, is_fixed=True), batch_size=2, shuffle=True)
val_dl = DataLoader(CustomDataset(
    filenames=iteration.valid_set, classes=classes, MAX_LEN=MAX_LEN, is_fixed=True), batch_size=2, shuffle=True)

# Checking dataloader

# Model parameters
input_dim = 31
hidden_dim = 256
layer_dim = 3       # Number of LSTM layers
output_dim = 8

# Training parameters
lr = 0.05
n_epochs = 100
best_accuracy = 0
patience, trials = 20, 0


log_folder = 'runs/scaled_original'

model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
model = model.cuda()

# Default loss function - CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
# Default optimizer - RMSprop
opt = torch.optim.RMSprop(model.parameters(), lr=lr)

# VISUALIZING MODEL ARCHITECTURE
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# summary(LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim),
#         input_size=(1, 1438, 31), col_names=["kernel_size", "input_size", "output_size", "num_params"],
#         verbose=2)

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
