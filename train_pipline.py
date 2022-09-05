import torch
import numpy as np
from tqdm import tqdm


def train(model, n_epochs, train_loader, valid_loader, optimizer, criterion, batch_size, ckpt_name):
    # Storage variable declaration
    train_acc_his, valid_acc_his = [], []
    train_losses_his, valid_losses_his = [], []

    for epoch in range(1, n_epochs + 1):
        # keep track of training and validation loss
        train_losses, valid_losses = [], []
        train_correct, val_correct, train_total, val_total = 0, 0, 0, 0
        train_pred, train_target = torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)
        val_pred, val_target = torch.zeros(batch_size, 1), torch.zeros(batch_size, 1)
        count = 0
        count2 = 0
        print('running epoch: {}'.format(epoch))

        # train the model #
        model.train()
        for data, target in tqdm(train_loader):
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss, criterion: loss_fnc
            loss = criterion(output, target)
            # calculate accuracy
            pred = output.data.max(dim=1, keepdim=True)[1]
            train_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            train_total += data.size(0)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_losses.append(loss.item() * data.size(0))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            if count == 0:
                train_pred = pred
                train_target = target.data.view_as(pred)
                count = count + 1
            else:
                train_pred = torch.cat((train_pred, pred), 0)
                train_target = torch.cat((train_target, target.data.view_as(pred)), 0)


        # validate the model
        model.eval()

        # tells PyTorch not to calculate the gradients
        with torch.no_grad():
            for data, target in tqdm(valid_loader):
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # calculate accuracy
                pred = output.data.max(dim=1, keepdim=True)[1]
                val_correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
                val_total += data.size(0)
                valid_losses.append(loss.item() * data.size(0))
                if count2 == 0:
                    val_pred = pred
                    val_target = target.data.view_as(pred)
                    count2 = count + 1
                else:
                    val_pred = torch.cat((val_pred, pred), 0)
                    val_target = torch.cat((val_target, target.data.view_as(pred)), 0)

        # calculate average losses
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)

        # calculate average accuracy
        train_acc = train_correct / train_total
        valid_acc = val_correct / val_total

        train_acc_his.append(train_acc)
        valid_acc_his.append(valid_acc)
        train_losses_his.append(train_loss)
        valid_losses_his.append(valid_loss)
        # print training/validation statistics
        print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            train_loss, valid_loss))
        print('\tTraining Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(
            train_acc, valid_acc))

        # Setting Check point
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'valid_acc': valid_acc,
                    'valid_loss': valid_loss,
                    }, ckpt_name)

    return train_acc_his, valid_acc_his, train_losses_his, valid_losses_his, model