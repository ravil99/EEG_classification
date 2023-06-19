import torch
import shutil
from torch import nn
from torch.nn import functional as F


def save_ckp(state, is_best, checkpoint_dir, best_model_dir, model_name):
    f_path = f"{checkpoint_dir}/checkpoint.pt"
    if is_best:
        f_path = f"{best_model_dir}/{model_name}_best_model.pt"
    torch.save(state, f_path)


def train(model, train_dl, val_dl, n_epochs, patience, opt, lr, criterion, writer, best_accuracy, model_name, checkpoint_dir, model_dir):
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
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict()
            }
            is_best = True
            save_ckp(checkpoint, is_best, checkpoint_dir,
                     model_dir, model_name)
            # torch.save(model.state_dict(), f'{model_name}_best.pth')
            print(
                f'Epoch {epoch} best model saved with Accuracy: {best_accuracy}')
        else:
            trials += 1
            is_best = False

            # if trials >= patience:
            #     print(f'Early stopping on epoch {epoch}')
            #     break

    print('The training is finished! Restoring the best model weights')
    writer.flush()
