#%% 
import torch 
from torch import nn
import numpy as np
from dataloaders import get_top_dataloaders, get_midi_dataloaders
import logging
from tqdm import tqdm
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

class trainer():
    def __init__(self, model, model_type, task_type, batch_size, nb_epochs, T=None, device=device) -> None:
        self.T = T
        self.model = model
        self.model = self.model.to(device).float()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.loss_printer_step = 1000
        self.nb_epochs = nb_epochs
        assert model_type in ["RNN", "LSTM", "GRU"], "model_type ({}) must be either RNN, LSTM or GRU".format(model_type)
        self.model_type = model_type
        assert task_type in ["top", "midi"], "task_type ({}) must be either top or midi".format(model_type)
        self.task_type = task_type

        # dataloaders and losses
        if self.task_type == "top":
            assert T is not None, "T is not defined"
            self.train_loader, self.eval_loader, self.test_loader = get_top_dataloaders(batch_size, T=T)
            self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
        if self.task_type == "midi":
            self.train_loader, self.eval_loader, self.test_loader = get_midi_dataloaders(batch_size)
            self.criterion = nn.CrossEntropyLoss()

        self.max_iterations = len(self.train_loader)
        
        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.5)

        # logs
        logs_file_name = "logs_top_{}_T{}_{}.txt".format(self.model_type, self.T, self.timestamp)
        self.logger, self.file_handler = self.get_log("logs", logs_file_name)

        # lists for tracking
        self.losses, self.norms_10, self.norms_50, self.norms_90, self.eigenvalues = [], [], [], [], []

    def train_1epoch(self, epoch_idx):
        if not(self.model.training): self.model.train()

        dataloader = iter(self.train_loader)

        states_seq = None

        for i, (x, target) in enumerate(tqdm(dataloader)):
            start_time = time.time()
            x = x.to(device)
            target = target.to(device)

            self.optimizer.zero_grad()

            output, states_seq = self.model(x)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.losses.append(loss.item())

            if i in [int(self.max_iterations/10), 5*int(self.max_iterations/10), 9*int(self.max_iterations/10)]:
                self.compute_grads_eigen(x, states_seq)

            ## Logs data
            if i % self.loss_printer_step == 0:
                duration = str(int(1000*(time.time()-start_time)))
                self.logger.info("Epoch {} - step {} lasted {}ms - loss {}".format(epoch_idx, i, duration, loss.item()))
    
    def eval_1epoch(self, epoch_idx):
        if self.model.training: self.model.eval()
        start_time = time.time()

        dataloader = iter(self.eval_loader)
        val_losses = list()

        with torch.no_grad():
            for x, target in dataloader:
                x = x.to(device)
                target = target.to(device)
                output, _ = self.model(x)
                val_loss = self.criterion(output, target)
                val_losses.append(val_loss.item())

        ## Logs data
        duration = str(int(1000*(time.time()-start_time)))
        self.logger.info("Epoch {} - Evaluation lasted {}ms - vl_loss {}".format(epoch_idx, duration, val_loss.item()))

        # return val_losses
        return val_losses
    
    def train(self):
        self.logger.info("Training with T={} - started at {}".format(self.T, self.timestamp))

        # for epoch_idx in tqdm(range(self.nb_epochs)):
        for epoch_idx in range(self.nb_epochs):
            epoch_start_time = time.time()
            self.logger.info("------ Epoch {} ------".format(epoch_idx))

            # train on one epoch 
            self.model.train(True)
            self.train_1epoch(epoch_idx)

            # eval after the epoch
            self.model.eval()
            val_losses = self.eval_1epoch(epoch_idx)
            
            # report data
            duration = str(int(time.time() - epoch_start_time))
            self.logger.info("Epoch {} lasted {}s - Avg training loss {} - Avg val loss {}".format(epoch_idx, duration, np.mean(self.losses), np.mean(val_losses)))
            self.logger.info("----------------------")

        # plots figures
        self.plot_magnitude()

        # save models
        model_path = 'trained_models/top_{}_T{}_{}.pth'.format(self.model_type, self.T, self.timestamp)
        torch.save(self.model.state_dict(), model_path)

        self.logger.info("End of training of top_{}_T{}_{}".format(self.model_type, self.T, self.timestamp))
        self.file_handler.close()
    
    def compute_grads_eigen(self, x, states_seq):
        T = x.size(1)
        if self.model_type == "RNN":
            dW = self.model.rnn.weight_hh_l0.grad
        else:
            dW = self.model.W_h.grad
        all_derivatives = self.model.compute_derivatives(x, states_seq)
        dxt_dxk_10 = self.model.compute_dxt_dxk(int(T/10), 1, all_derivatives)
        dxt_dxk_50 = self.model.compute_dxt_dxk(5*int(T/10), 1, all_derivatives)
        dxt_dxk_90 = self.model.compute_dxt_dxk(9*int(T/10), 1, all_derivatives) 
        try:
            if self.model_type == "RNN":
                eigen = torch.abs(torch.linalg.eigvals(dW))
            else:
                HS = self.model.hidden_dim
                eigen = torch.abs(torch.linalg.eigvals(dW[:,HS:HS*2]))
        except:
            print("unable to compute the eigenvalue")
            # print(dW.shape)
            # print(dW)
        
        self.eigenvalues.append(torch.max(eigen).item())
        self.norms_10.append(torch.norm(dxt_dxk_10).item())
        self.norms_50.append(torch.norm(dxt_dxk_50).item())
        self.norms_90.append(torch.norm(dxt_dxk_90).item())

    def plot_magnitude(self):
        indexes = np.arange(1, 4*self.nb_epochs)
        bar_width = 0.3
        plt.bar(np.array(indexes) - bar_width, np.log10(self.norms_10), width=bar_width, label='t=T/10', edgecolor='black')
        plt.bar(np.array(indexes), np.log10(self.norms_50), width=bar_width, label='t=T/2', edgecolor='black')
        plt.bar(np.array(indexes) + bar_width, np.log10(self.norms_90), width=bar_width, label='t=9T/10', edgecolor='black')

        bar_positions = np.arange(1, 4)
        for i, v1, v2, v3 in zip(bar_positions, self.norms_10, self.norms_50, self.norms_90):
            if v1 == 0.0 or np.isnan(v1):
                plt.bar(i - bar_width, -20, width=bar_width, color='black', edgecolor='white')
            if  np.isinf(v1):
                plt.bar(i - bar_width, 20, width=bar_width, color='black', edgecolor='white')
            if v2 == 0.0 or np.isnan(v2):
                plt.bar(i, -20, width=bar_width, color='black', edgecolor='white')
            if  np.isinf(v2):
                plt.bar(i, 20, width=bar_width, color='black', edgecolor='white')
            if v3 == 0.0 or np.isnan(v3):
                plt.bar(i + bar_width, -20, width=bar_width, color='black', edgecolor='white')
            if  np.isinf(v3):
                plt.bar(i + bar_width, 20, width=bar_width, color='black', edgecolor='white')

        plt.xticks(bar_positions, [r'$ \lambda $ = ' + '{:.2e}'.format(eigen) for eigen in self.eigenvalues])
        plt.xlabel('Iteration')
        plt.ylabel('log_10(||dx_t/dx_k||)')
        if self.model_type == "LSTM":
            plt.title('Order of magnitude of ||dc_t/dc_k|| for ' + self.model_type)
        else:
            plt.title('Order of magnitude of ||dh_t/dh_k|| for ' + self.model_type)
        plt.axhline(y=0, color='black')
        plt.legend()
        plt.show()
  
    def get_log(self, path, file):
        log_file = os.path.join(path, file)
        if not os.path.isfile(log_file):
            open(log_file, "w+").close()

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger, file_handler




#%% Experiments 

from top_models import top_RNN, top_LSTM, top_GRU
from midi_models import midi_RNN, midi_LSTM, midi_GRU

device = "cpu"
T=100
RNN_top = top_RNN(input_size=6, hidden_dim=50, output_size=2, nonlinearity='tanh')
LSTM_top = top_LSTM(input_size=6, hidden_dim=50, output_size=2)
GRU_top = top_GRU(input_size=6, hidden_dim=50, output_size=2)

RNN_midi = midi_RNN(input_size=128, hidden_dim=256, output_size=128, nonlinearity='tanh')
LSTM_midi = midi_LSTM(input_size=128, hidden_dim=256, output_size=128)
GRU_midi = midi_GRU(input_size=128, hidden_dim=256, output_size=128)

# models = [RNN_top, RNN_midi, LSTM_top, LSTM_midi, GRU_top, GRU_midi]
# model_types = ["RNN", "RNN", "LSTM", "LSTM", "GRU", "GRU"]
# task_types = ["top", 'midi']

trainer_exp = trainer(RNN_top, model_type='RNN', task_type='top', T=T, batch_size=16, nb_epochs=1, device="cpu")
trainer_exp.train()