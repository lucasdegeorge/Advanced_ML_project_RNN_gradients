#%% 
import torch
from torch import nn
import math


class top_RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, nonlinearity='tanh', n_layers=1):
        super(top_RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.nonlinearity = nonlinearity
        self.output_size = output_size
        
        assert nonlinearity in ['tanh', 'relu'], "nonlinearity ({}) must be either 'tanh' or 'relu'".format(nonlinearity)
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, nonlinearity=nonlinearity)
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        """x is of shape (batch, sequence, hidden_size) """
        h0 = self.init_hidden(x.size(0))
        states_seq, _ = self.rnn(x, h0)   # states_seq contains all h_t for all t
        output = self.fc(states_seq[:,-1,:]) 
        # print("output", output.shape)
        return output, states_seq
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
    
    def compute_derivatives(self, x, states_seq):
        """ Compute the derivative dhi_dhi_1 for all i. 
            states_seq has shape (batch, T, hidden_size). x is useless here.
            res has shape (batch, T, feature, feature)
            W_rec is notation from "On the diffculty of training Recurrent Neural Networks" paper """
        
        W_rec_T = self.rnn.weight_hh_l0.T
        # diag(sigma'(x_k))
        if self.nonlinearity == 'tanh':
            diags = 1 - torch.square(torch.tanh(states_seq))
            diags = torch.diag_embed(diags, dim1=2, dim2=3)
        if self.nonlinearity == 'relu':
            diags = torch.where(states_seq > 0, torch.tensor(1.0), torch.tensor(0.0))
            diags = torch.diag_embed(diags, dim1=2, dim2=3)
        
        return torch.matmul(W_rec_T.unsqueeze(0), diags)
    
    def compute_dxt_dxk(self, t, k, all_derivatives):
        dxt_dxk = torch.prod(all_derivatives[:, k:t+1, :, :], dim=1)
        return dxt_dxk
    
    def compute_omega(self, W_rec, states_seq):
        """ Computes the sum of the omega_k over k. 
            Formula given by Equation 10 of Section 3.3 of On the diffiulty of training Recurrent Neural Networks. """
        
        # diags = diag(sigma'(x_k)) of size (batch, T, feature, feature)
        if self.nonlinearity == 'tanh':
            diags = 1 - torch.square(torch.tanh(states_seq))
            diags = torch.diag_embed(diags, dim1=2, dim2=3)
        if self.nonlinearity == 'relu':
            diags = torch.where(states_seq > 0, torch.tensor(1.0), torch.tensor(0.0))
            diags = torch.diag_embed(diags, dim1=2, dim2=3)
        
        dE_dxk1 = torch.zeros_like(states_seq.grad)  # dE/dx_{k+1} 
        dE_dxk1[:,:-1] = states_seq.grad[:,1:]  # [bs, T, feature]
        dxk1_dxk = torch.matmul(W_rec.T.unsqueeze(0), diags)  # [bs, T, feature, feature]
        omega = torch.matmul(dxk1_dxk, dE_dxk1.unsqueeze(-1)).squeeze(-1)  # dE/dx_{k+1} * W_rec.T diag(sigma'(x_k))  size: [bs, T, feature]
        omega = torch.square((torch.norm(omega, dim=-1) - torch.norm(dE_dxk1, dim=-1)) / torch.norm(dE_dxk1, dim=-1))  
        # print(omega)

        return omega



class top_LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super().__init__()
        self.input_sz = input_size
        self.hidden_dim = hidden_dim
        self.W_i = nn.Parameter(torch.Tensor(input_size, hidden_dim * 4))  # W = [W_i, W_f, W_c, W_o]
        self.W_h = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 4))
        self.init_weights()
        self.fc = nn.Linear(hidden_dim, output_size)
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, init_states=None):
        """x is of shape (batch, sequence, hidden_size). g_t is also known as c_tilde_t"""
        batch_size, T, _= x.size()
        HS = self.hidden_dim
        states_seq = torch.zeros(T, 6, batch_size, HS).to(x.device)  # h_t, c_t, i_t, f_t, g_t, o_t

        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, HS).to(x.device), 
                        torch.zeros(batch_size, HS).to(x.device))
        else:
            h_t, c_t = init_states
         
        for t in range(T):
            x_t = x[:, t, :]
            gates = x_t @ self.W_i + h_t @ self.W_h + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]), # context
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            states_seq[t, :, :, :] = torch.stack((h_t, c_t, i_t, f_t, g_t, o_t), dim=0)
        
        output = self.fc(states_seq[-1, 0, :, :])

        # from shape (T, 6, batch_size, hidden_size) to (batch, 6, T, hidden_size)
        states_seq = states_seq.transpose(0, 2).contiguous()
        
        return output, states_seq
    
    def compute_derivatives(self, x, states_seq):
        """ Compute the derivative dci_dci_1 for all i. 
            states_seq has shape (batch, 6, T, hidden_size). x has shape (batch, T, hidden_size)
            res has shape (batch, T, T, T)
            A_t, B_t, C_t, D_t are notation from the report """

        batch_size, T, _= x.size()
        HS = self.hidden_dim

        gates = x @ self.W_i + states_seq[:, 0, :, :] @ self.W_h + self.bias   # (batch, T, hidden_size)

        states_minus_1 = torch.zeros(batch_size, 3, T, HS)  # h, c and o (we only need h_{t-1}, c_{t-1} and o_{t-1})
        
        states_minus_1[:, 0, 1:, :] = states_seq[:, 0, :, :][:, :-1, :].detach().clone()  # h
        states_minus_1[:, 1, 1:, :] = states_seq[:, 1, :, :][:, :-1, :].detach().clone()  # c
        states_minus_1[:, 2, 1:, :] = states_seq[:, 5, :, :][:, :-1, :].detach().clone()  # o

        tanh_prime_c_t1 = 1 - torch.square(torch.tanh(states_minus_1[:, 1, :, :]))  # tanh'(c_{t-1})
        o_tanh_prime = states_minus_1[:, 2, :, :] * tanh_prime_c_t1  # o_{t-1} * tanh'(c_{t-1})
        o_tanh_prime = torch.diag_embed(o_tanh_prime, dim1=2, dim2=3)  

        A_t = torch.diag_embed(torch.where(gates[:, :, HS:HS*2] > 0, torch.tensor(1.0), torch.tensor(0.0)), dim1=2, dim2=3)  # diag(sigma'(W_f[h_{t-1}, x]))
        A_t = torch.matmul(A_t, self.W_h[:,HS:HS*2])  # W_f
        A_t = torch.matmul(A_t, o_tanh_prime)  # o_{t-1} * tanh'(c_{t-1})
        A_t = torch.matmul(A_t, torch.diag_embed(states_minus_1[:, 1, :, :], dim1=2, dim2=3))  # diag(c_{t-1})

        B_t = torch.diag_embed(states_seq[:, 3, :, :], dim1=2, dim2=3)  # diag(f_t)

        C_t = torch.diag_embed(torch.where(gates[:, :, :HS] > 0, torch.tensor(1.0), torch.tensor(0.0)), dim1=2, dim2=3)  # diag(sigma'(W_i[h_{t-1}, x]))
        C_t = torch.matmul(C_t, self.W_h[:,:HS])  # W_i
        C_t = torch.matmul(C_t, o_tanh_prime)  # o_{t-1} * tanh'(c_{t-1})
        C_t = torch.matmul(C_t, torch.diag_embed(states_seq[:, 4, :, :], dim1=2, dim2=3))  # diag(g_t)

        D_t = torch.diag_embed(torch.where(gates[:, :, HS*2:HS*3] > 0, torch.tensor(1.0), torch.tensor(0.0)), dim1=2, dim2=3)  # diag(sigma'(W_c[h_{t-1}, x]))
        D_t = torch.matmul(D_t, self.W_h[:,HS*2:HS*3])  # W_c
        D_t = torch.matmul(D_t, o_tanh_prime)  # o_{t-1} * tanh'(c_{t-1})
        D_t = torch.matmul(D_t, torch.diag_embed(states_minus_1[:, 2, :, :], dim1=2, dim2=3))  # diag(i_t)
       
        return A_t + B_t + C_t + D_t
    
    def compute_dxt_dxk(self, t, k, all_derivatives):
        dxt_dxk = torch.prod(all_derivatives[0, k:t+1], dim=0)
        return dxt_dxk


class top_GRU(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super().__init__()
        self.input_sz = input_size
        self.hidden_dim = hidden_dim
        self.W_i = nn.Parameter(torch.Tensor(input_size, hidden_dim * 3))  # W = [W_r, W_z, W_n]
        self.W_h = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 3))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim * 3))
        self.init_weights()
        self.fc = nn.Linear(hidden_dim, output_size)
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, init_h_t=None):
        """x is of shape (batch, sequence, hidden_size) """
        batch_size, T, _= x.size()
        HS = self.hidden_dim
        states_seq = torch.zeros(T, 4, batch_size, HS).to(x.device)  # h_t, r_t, z_t, n_t

        if init_h_t is None:
            h_t = torch.zeros(batch_size, HS).to(x.device)
        else:
            h_t = init_h_t
         
        for t in range(T):
            x_t = x[:, t, :]
            gates = x_t @ self.W_i + h_t @ self.W_h + self.bias
            r_t, z_t = (
                torch.sigmoid(gates[:, :HS]), # reset
                torch.sigmoid(gates[:, HS:HS*2]), # update
            )
            n_t = torch.tanh(gates[:, HS*2:] + r_t * gates[:, HS*2:])  # new
            h_t = (1-z_t) * n_t + z_t * h_t

            states_seq[t, :, :, :] = torch.stack((h_t, r_t, z_t, n_t), dim=0)
        
        output = self.fc(states_seq[-1, 0, :, :])

        # from shape (T, 4, batch_size, hidden_size) to (batch, 4, T, hidden_size)
        states_seq = states_seq.transpose(0, 2).contiguous()
        
        return output, states_seq
    
    def compute_derivatives(self, x, states_seq):
        """ Compute the derivative dhi_dhi_1 for all i. 
            states_seq has shape (batch, 6, T, hidden_size). x has shape (batch, T, hidden_size)
            res has shape (batch, T, hidden_size, hidden_size)
            A_t, B_t, C_t, D_t are notation from the report. """

        batch_size, T, _= x.size()
        HS = self.hidden_dim

        gates = x @ self.W_i + states_seq[:, 0, :, :] @ self.W_h + self.bias   # (batch, T, hidden_size)

        h_minus_1 = torch.zeros(batch_size, T, HS)  # we only need h_{t-1} 
        
        h_minus_1[:, 1:, :] = states_seq[:, 0, :, :][:, :-1, :].detach().clone()  # h

        A_t = -1 * torch.diag_embed(torch.where(gates[:, :, HS:HS*2] > 0, torch.tensor(1.0), torch.tensor(0.0)), dim1=2, dim2=3)  # - diag(sigma'(W_z[x,h]))
        A_t = torch.matmul(A_t, self.W_h[:,HS:HS*2])  # W_hz
        A_t = torch.matmul(A_t, torch.diag_embed(states_seq[:, 2, :, :], dim1=2, dim2=3))  # diag(n_t)

        B_t = torch.diag_embed(1-gates[:, :, HS:HS*2], dim1=2, dim2=3)  # diag(1-z_t)
        tanh_prime_gates = 1 - torch.square(torch.tanh(gates[:, :, HS*2:]))  # tanh'(W_n[x,h])
        B_t = torch.matmul(B_t, torch.diag_embed(tanh_prime_gates, dim1=2, dim2=3))  # diag(tanh'(W_n[x,h]))
        B_t = torch.matmul(B_t, torch.diag_embed(states_seq[:, 1, :, :], dim1=2, dim2=3))  # diag(r_t)
        B_t = torch.matmul(B_t, self.W_h[:, HS*2:])  # W_hn

        C_t = torch.diag_embed(torch.where(gates[:, :, HS:HS*2] > 0, torch.tensor(1.0), torch.tensor(0.0)), dim1=2, dim2=3)  # diag(sigma'(W_z[x,h]))
        C_t = torch.matmul(C_t, self.W_h[:,HS:HS*2])  # W_hz
        C_t = torch.matmul(C_t, torch.diag_embed(h_minus_1, dim1=2, dim2=3))  # diag(h_{t-1})

        D_t = torch.diag_embed(states_seq[:, 2, :, :], dim1=2, dim2=3)  # diag(z_t)

        return A_t + B_t + C_t + D_t
    
    def compute_dxt_dxk(self, t, k, all_derivatives):
        dxt_dxk = torch.prod(all_derivatives[:, k:t+1, :, :], dim=1)
        return dxt_dxk



