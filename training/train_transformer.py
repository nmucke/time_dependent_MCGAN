import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import data_handling.data_handling_utils as utils


class TrainTransformer():
    def __init__(self,
                 transformer,
                 transformer_optimizer,
                 scheduler=None,
                 save_string='transformer',
                 n_epochs=100,
                 memory=100,
                 device='cpu'):

        self.device = device
        self.model = transformer
        self.optimizer = transformer_optimizer

        self.scheduler = scheduler


        self.n_epochs = n_epochs
        self.save_string = save_string

        self.model.train(mode=True)

        self.memory = memory

        self.loss_function = nn.MSELoss()
        #self.loss_function = nn.L1Loss()


    def train(self, dataloader, temporal_batch_size=32):
        """Train generator and critic"""

        loss_list = []
        for epoch in range(1, self.n_epochs + 1):
            self.epoch = epoch

            # Train one step
            loss = self.train_epoch(dataloader=dataloader,
                                    temporal_batch_size=temporal_batch_size)

            if self.scheduler is not None:
                self.scheduler.step()

            # Save loss
            loss_list.append(loss)

            # Save generator and critic weights
            torch.save({
                'transformer_state_dict': self.model.state_dict(),
                'transformer_optimizer_state_dict': self.optimizer.state_dict(),
                }, self.save_string)

        # Save generator and critic weights

        torch.save({
            'transformer_state_dict': self.model.state_dict(),
            'transformer_optimizer_state_dict': self.optimizer.state_dict(),
            }, self.save_string)

        self.model.eval()

        return loss_list

    def train_epoch(self, dataloader, temporal_batch_size):
        """Train generator and critic for one epoch"""

        total_loss = 0
        progress_bar = tqdm(enumerate(dataloader),
                            total=len(dataloader.dataset) // dataloader.batch_size,
                            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for bidx, (features, targets) in progress_bar:
            loss = 0
            shuffled_ids = np.arange(0,features.shape[1])
            np.random.shuffle(shuffled_ids)
            features[:, shuffled_ids] = features[:, shuffled_ids]
            targets[:, shuffled_ids] = targets[:, shuffled_ids]
            for batch, i in enumerate(range(0, features.shape[1]-1, temporal_batch_size)):
                batch_features, batch_targets = utils.get_batch(
                    features=features,
                    targets=targets,
                    time_step=i,
                    batch_size=temporal_batch_size
                )

                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)

                loss += self.train_step(batch_features, batch_targets)

            progress_bar.set_postfix({"Loss": loss/batch,
                                      'Epoch': self.epoch,
                                      'of': self.n_epochs})
            total_loss += loss/batch
        return total_loss

    def train_step(self, features, targets):
        self.optimizer.zero_grad()
        output = self.model(features)
        loss = self.loss_function(output, targets[-1:, :, :])
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.7)
        self.optimizer.step()

        return loss.detach().item()


