import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.conf import device


def train_model(model, hsi_data, dataloader, loss, num_epochs, n_sources, need_plot=False):
    print(f"TRAINING WILL BE DONE ON {device}")
    loss_train = []
    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    
    model.to(device).train()

    for epoch in range(num_epochs):
        print(' Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 40)

        print("Learning rate:", optimizer.param_groups[0]['lr'])

        epoch_samples = 0
        running_loss = 0

        for batch in tqdm(dataloader):
            inputs_X = batch['X']

            if 'A' in batch and 'S' in batch:
                inputs_A = batch['A'].to(device)
                inputs_S = batch['S'].to(device)
            else:
                inputs_A, inputs_S = None, None

            optimizer.zero_grad()

            # forward
            X_pred, A_pred, S_pred = model(inputs_X.to(device))

            loss_value = loss(X_pred, A_pred, S_pred, inputs_X, inputs_A, inputs_S)

            loss_value.backward()
            optimizer.step()

            # statistics
            epoch_samples += inputs_X.size(0)
            running_loss += loss_value.item() * inputs_X.size(0)
        scheduler.step()

        epoch_loss = running_loss / epoch_samples

        loss_train.append(epoch_loss)

        if need_plot and (epoch % 5 == 0):
            _plot_while_training(model, hsi_data, n_sources)

        print('Loss: {:.4f}'.format(epoch_loss))

    return model, loss_train


def _plot_while_training(model, hsi_data, n_sources):
    model.eval()
    pred, endmemb, abund = model(torch.Tensor(hsi_data[None, :, :, :]))
    pred, endmemb, abund = pred.detach().numpy(), endmemb.detach().numpy(), abund.detach().numpy()

    plt.figure(figsize=(6, 4))
    for ii in range(n_sources):
        ax = plt.subplot(2, n_sources // 2, ii + 1)
        ax.imshow(abund[0, ii, :, :])
    plt.show()

    plt.figure(figsize=(6, 4))
    for i in range(n_sources):
        ax = plt.subplot(2, n_sources // 2, i + 1)
        ax.plot(endmemb[:, i])
    plt.show()
    model.train()
