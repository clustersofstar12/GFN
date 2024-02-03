import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

class AssignLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AssignLayer, self).__init__()
        self.w = nn.Parameter(torch.empty(size=(input_dim, output_dim)))
        #nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        '''self.fu = nn.Sequential(
            nn.BatchNorm1d(kc),
            nn.LeakyReLU(0.2),)'''
        #self.kc = kc
        #self.n_view = n_view

    def forward(self, f, adj):
        f = F.dropout(f, 0.2, training=self.training)
        f = torch.matmul(adj, f)
        f = torch.matmul(f, self.w)  # [B, k, k1]
        f = f + self.bias
        #s = s.view(-1, self.kc)
        #s = self.fu(s)
        #s = F.softmax(s, dim=-1)
        #s = s.view(-1, self.n_view, self.kc)
        return f


class Assign(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, out_dim):
        super(Assign, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.out_dim = out_dim
        self.conv_first = AssignLayer(input_dim=input_dim, output_dim=hidden_dim)
        self.conv_block = nn.ModuleList(
                [AssignLayer(input_dim=hidden_dim, output_dim=hidden_dim)for _ in range(3)])
        self.conv_last = AssignLayer(input_dim=hidden_dim, output_dim=embedding_dim)

        self.MP = self.MPlayer(out_dim=out_dim)
        self.act = nn.LeakyReLU(0.2)

        self.fusion = nn.Sequential(
            nn.Linear(4 * self.hidden_dim + self.embedding_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2),)


        for m in self.modules():
            if isinstance(m, AssignLayer):
                m.w.data = init.xavier_uniform_(m.w.data, gain=nn.init.calculate_gain('relu'))
                #m.w.data = init.xavier_normal_(m.w.data, gain=1)
                #m.w.data = init.xavier_uniform_(m.w.data, gain=1)
                #m.w.data = init.constant_(m.w.data, 1)
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def apply_bn(self, x):
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def MPlayer(self, out_dim):
        '''if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)'''
        mp_layers = []
        mp_input_dim = 4 * self.hidden_dim + self.embedding_dim
        mp_hidden_dims = [512, 256, 128]

        mp_dim = 0
        for mp_dim in mp_hidden_dims:
            mp_layers.append(nn.Linear(mp_input_dim, mp_dim))
            mp_layers.append(nn.BatchNorm1d(mp_dim))
            mp_layers.append(nn.LeakyReLU(0.2))
            mp_input_dim = mp_dim
        mp_layers.append(nn.Linear(mp_dim, out_dim))
        mp_model = nn.Sequential(*mp_layers)
        return mp_model

    def forward(self, x, adj):
        n_views = x.size()[1]
        x = self.conv_first(x, adj)
        x = self.act(x)
        x = self.apply_bn(x)
        x_all = [x]
        #out_all = []
        #out, _ = torch.max(x, dim=1)
        #out_all.append(out)
        for i in range(len(self.conv_block)):
            x = self.conv_block[i](x, adj)
            x = self.act(x)
            x = self.apply_bn(x)
            x_all.append(x)
        x = self.conv_last(x, adj)
        x_all.append(x)
        s = torch.cat(x_all, dim=2)
        s = s.view(-1, 4 * self.hidden_dim + self.embedding_dim)
        #s = self.MP(s)
        s = self.fusion(s)
        s = s.view(-1, n_views, self.out_dim)
        s = F.softmax(s, dim=-1)
        return s








