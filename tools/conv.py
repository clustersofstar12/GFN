import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

class ConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvLayer, self).__init__()
        self.w = nn.Parameter(torch.empty(size=(input_dim, output_dim)))
        #nn.init.xavier_uniform_(self.w.data, gain=1.414)
        self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        self.fu = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),)
        #self.kc = kc
        #self.n_view = n_view

    def forward(self, f, adj):
        #f = F.dropout(f, 0.2, training=self.training)
        _, N, _ = f.size()
        f = torch.matmul(adj, f)
        f = torch.matmul(f, self.w)  # [B, k, k1]
        f = f + self.bias
        f = f.view(-1, 512)
        f = self.fu(f)
        f = f.view(-1, N, 512)
        #s = s.view(-1, self.kc)
        #s = self.fu(s)
        #s = F.softmax(s, dim=-1)
        #s = s.view(-1, self.n_view, self.kc)
        return f


class Conv(nn.Module):
    def __init__(self, input_dim, out_dim, s_dim, num_layers):
        super(Conv, self).__init__()

        #self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.s_dim = s_dim
        self.num_layers = num_layers
        #self.conv_first = ConvLayer(input_dim=input_dim, output_dim=hidden_dim)
        self.conv_block = nn.ModuleList(
                [ConvLayer(input_dim=input_dim, output_dim=out_dim)for _ in range(num_layers)])
        #self.conv_last = ConvLayer(input_dim=hidden_dim, output_dim=out_dim)

        #self.MP = self.MPlayer(out_dim=out_dim)
        self.act = nn.LeakyReLU(0.2)
        self.fusion = nn.Sequential(
            nn.Linear(num_layers * out_dim, s_dim),
            nn.BatchNorm1d(s_dim),
            nn.LeakyReLU(0.2),)

        for m in self.modules():
            if isinstance(m, ConvLayer):
                m.w.data = init.xavier_uniform_(m.w.data, gain=nn.init.calculate_gain('relu'))
                #m.w.data = init.xavier_normal_(m.w.data, gain=1)
                #m.w.data = init.xavier_uniform_(m.w.data, gain=1)
                #m.w.data = init.constant_(m.w.data, 1)
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def apply_bn(self, x):
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, x, adj):
        '''x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)'''
        n_views = x.size()[1]
        out_all = []
        s_all = []
        #out, _ = torch.max(x, dim=1)
        #out_all.append(out)
        for i in range(len(self.conv_block)):
            x = self.conv_block[i](x, adj)
            #x = self.act(x)
            #x = self.apply_bn(x)
            s_all.append(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            #if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        s = torch.cat(s_all, dim=2)
        s = s.view(-1, self.num_layers * self.out_dim)
        s = self.fusion(s)
        s = s.view(-1, n_views, self.s_dim)
        s = F.softmax(s, dim=-1)
        '''x = self.conv_last(x, adj)
        #x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)'''
        '''if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)'''
        output = torch.cat(out_all, dim=1)
        return x, output, s