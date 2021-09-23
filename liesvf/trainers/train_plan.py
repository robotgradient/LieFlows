from __future__ import print_function
import torch
from torch.utils.data import DataLoader, Dataset
from liesvf.utils import save_in_torch_file
import time
import copy

def regression_trainer(model, loss_fn, optimizer, dataset, device,
                       n_epochs=500, batch_size=100, shuffle=True,
                       clip_gradient=True, clip_value_grad=0.1,
                       clip_weight=False, clip_value_weight=2,
                       vis_freq=10, vis_fn=None, model_save_file=None,
                       log_freq=5, logger=None, loss_clip=1e3, stop_threshold=float('inf')):

    ## Build Dataset##
    params = {'batch_size': batch_size, 'shuffle': True}
    dataloader = DataLoader(dataset, **params)
    n_trjs = len(dataset.x)
    n_samples = dataset.len

    ts = time.time()

    best_train_loss = float('inf')
    best_train_epoch = 0
    best_model = model

    model.train()
    for epoch in range(n_epochs):
        ## Training ##
        train_loss = 0
        for x_batch, y_batch in dataloader:

            x_batch = x_batch.to(device)
            x_batch.requires_grad = True
            y_batch = y_batch.to(device)


            loss = loss_fn(model, x_batch, y_batch)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            if clip_gradient:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip_value_grad
                )
            optimizer.step()

        train_loss = float(batch_size) / float(n_samples) * train_loss
        if train_loss < best_train_loss:
            best_train_epoch = epoch
            best_train_loss = train_loss
            best_model = copy.deepcopy(model)

            ## Save Model ##
            if model_save_file is not None:
                names = ['params']
                variables = [best_model.state_dict()]
                save_in_torch_file(model_save_file, names, variables)
            ################

            # report loss in command line and tensorboard every log_freq epochs
            if epoch % log_freq == (log_freq - 1):
                print('    Epoch [{}/{}]: current loss is {}, time elapsed {} second'.format(
                    epoch + 1, n_epochs,
                    train_loss,
                    time.time() - ts)
                )

                if logger is not None:
                    info = {'Training Loss': train_loss}

                    # log scalar values (scalar summary)
                    for tag, value in info.items():
                        logger.add_scalar(tag, value, epoch + 1)

                    # log values and gradients of the parameters (histogram summary)
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        logger.add_histogram(
                            tag,
                            value.data.cpu().numpy(),
                            epoch + 1
                        )
                        if value.grad is not None:
                            logger.add_histogram(
                                tag + '/grad',
                                value.grad.data.cpu().numpy(),
                                epoch + 1
                            )

        if epoch % vis_freq == (vis_freq - 1):
                if vis_fn is not None:
                    vis_fn(model)


    return best_model, best_train_loss

