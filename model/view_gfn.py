import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from model.Model import Model
from tools.utils import LocalGCN, GCN, AttentionMP, NonLocalMP
from tools.mincutpool import dense_mincut_pool, normalize, cutselfloop
from tools.diffpool import dense_diff_pool
from tools.assign import Assign
from tools.conv import Conv
from tools import meter


mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, requires_grad=False)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, requires_grad=False)

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

class SVCNN(Model):
    def __init__(self, name, nclasses=51, pretraining=True, cnn_name='resnet18'):
        super(SVCNN, self).__init__(name)
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.use_resnet = cnn_name.startswith('resnet')
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, requires_grad=False)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, requires_grad=False)

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net.fc = nn.Linear(512, self.nclasses)
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net.fc = nn.Linear(2048, self.nclasses)
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11_bn(pretrained=self.pretraining).features
                self.net_2 = models.vgg11_bn(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier

            self.net_2._modules['6'] = nn.Linear(4096, self.nclasses)

    def forward(self, x):
        if self.use_resnet:
            return self.net(x)
        else:
            y = self.net_1(x)
            return self.net_2(y.view(y.shape[0], -1))

class view_GFN(Model):

    def __init__(self, name, model, nclasses=51, cnn_name='resnet18', num_views=12):
        super(view_GFN, self).__init__(name)
        self.classnames = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                           'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
                           'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
                           'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
                           'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, requires_grad=False)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, requires_grad=False)
        self.use_resnet = cnn_name.startswith('resnet')
        if self.use_resnet:
            self.net_1 = nn.Sequential(*list(model.net.children())[:-1])
            self.net_2 = model.net.fc
        else:
            self.net_1 = model.net_1
            self.net_2 = model.net_2
        if self.num_views == 20:
            phi = (1 + np.sqrt(5)) / 2
            vertices = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                        [0, 1 / phi, phi], [0, 1 / phi, -phi], [0, -1 / phi, phi], [0, -1 / phi, -phi],
                        [phi, 0, 1 / phi], [phi, 0, -1 / phi], [-phi, 0, 1 / phi], [-phi, 0, -1 / phi],
                        [1 / phi, phi, 0], [-1 / phi, phi, 0], [1 / phi, -phi, 0], [-1 / phi, -phi, 0]]
        elif self.num_views == 12:
            phi = np.sqrt(3)
            vertices = [[1, 0, phi/3], [phi/2, -1/2, phi/3], [1/2,-phi/2,phi/3],
                        [0, -1, phi/3], [-1/2, -phi/2, phi/3],[-phi/2, -1/2, phi/3],
                        [-1, 0, phi/3], [-phi/2, 1/2, phi/3], [-1/2, phi/2, phi/3],
                        [0, 1 , phi/3], [1/2, phi / 2, phi/3], [phi / 2, 1/2, phi/3]]

            #adj = np.zeros(shape=(12, 12))

        self.vertices = torch.tensor(vertices).cuda()
        #self.adj = adj
        self.kc1 = 8
        self.kc2 = 3

        #self.LocalGCN1 = LocalGCN(k=4, n_views=self.num_views)
        #self.NonLocalMP1 = NonLocalMP(n_view=8)
        #self.GCN1 = GCN(bias=True, n_view=self.kc1)
        #self.NonLocalMP2 = NonLocalMP(n_view=3)
        #self.GCN2 = GCN(bias=True, n_view=self.kc2)
        self.GCN1 = Conv(input_dim=512, out_dim=512, s_dim=4, num_layers=3)
        self.GCN2 = Conv(input_dim=512, out_dim=512, s_dim=2, num_layers=3)
        self.GCN3 = Conv(input_dim=512, out_dim=512, s_dim=1, num_layers=1)
        #self.View_selector1 = View_selector(n_views=self.num_views, sampled_view=self.num_views//2)
        #self.View_selector2 = View_selector(n_views=self.num_views//2, sampled_view=self.num_views//4)
        #self.AssignMatrix1 = AssignMatrix(kc=10, n_view=self.num_views)
        #self.AssignMatrix2 = AssignMat(kc=5, n_view=self.kc1)
        #self.AssignMatrix1 = Assign(input_dim=512, hidden_dim=512, embedding_dim=512, out_dim=8)
        #self.AssignMatrix2 = Assign(input_dim=512, hidden_dim=512, embedding_dim=512, out_dim=3)
        #self.AttentionMP1 = AttentionMP(dropout=0.2)
        #self.AttentionMP2 = AttentionMP(dropout=0.19)
        #self.AttentionMP3 = AttentionMP(dropout=0.21)

        self.cls1 = nn.Sequential(
            nn.Linear(512*14, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.cls2 = nn.Sequential(
            nn.Linear(512, self.nclasses)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)

        adj = np.ones(shape=(12, 12))
        adj = torch.tensor(adj).cuda()
        self.adj = adj.float()

        '''adj = adj.unsqueeze(0).repeat(y.shape[0], 1, 1)  # [B, V, V]
        self.adj = adj.float()'''

    def forward(self, x):
        #model_time = meter.TimeMeter(True)
        #model_time.reset()
        views = self.num_views
        y = self.net_1(x)
        y = y.view((int(x.shape[0] / views), views, -1))
        #vertices = self.vertices.unsqueeze(0).repeat(y.shape[0], 1, 1)
        #print(vertices.dtype)
        #vertices = vertices.float()
        #print(vertices.dtype)

        #index = self.LocalGCN1(y, vertices)  # y[B, V, C]
        #print(v.size())
        #print('v', v)
        #print(index.size())
        #print('index', index)
        #pooled_view1 = torch.max(y, 1)[0]
        #pool_view1 = torch.mean(y, dim=1, keepdim=False)

        # 求邻接矩阵
        #v = v.cpu()
        #index = index.cpu()
        #v1 = v.detach().numpy()
        #index = index.detach().numpy()
        '''adj = np.zeros(shape=(20, 20))
        for row in range(20):
            for column in range(20):
                #for id in range(4):
                    #if(column == index[row][id]):
                        #adj[row][column] = v1[row][id]
                adj[row][column] = 1.0'''
        #adj = np.ones(shape=(20, 20))
        #print('adj', adj)

        '''for row in range(adj.shape[0]):
            for column in range(adj.shape[1]):
                if(row > column):
                    adj[row][column] = (adj[row][column] + adj[column][row])/2
                    adj[column][row] = adj[row][column]'''

        '''a_time = meter.TimeMeter(True)
        a_time.reset()
        adj = torch.tensor(adj).cuda()
        print('atime %.6f' % a_time.value())'''
        adj = self.adj.unsqueeze(0).repeat(y.shape[0], 1, 1)  # [B, V, V]
        #adj = adj.float()
        #adj = normalize(adj)

        y, pooled_view1, s1 = self.GCN1(y, adj)

        '''pooled_view1 = torch.max(y, 1)[0]
        pool_view1 = torch.mean(y, dim=1, keepdim=False)'''

        #s1 = self.AssignMatrix1(y, adj)  # [B, V, k]
        #print('adj_grad', adj.requires_grad)
        #print('s1_grad', s1.requires_grad)
        #print('adj', adj)
        #print('s1', s1)

        #mincut_loss1, ortho_loss1 = dense_mincut_pool(y, adj, s1, mask=None)
        #link_loss1, ent_loss1 = dense_diff_pool(y, adj, s1, mask=None)

        z = torch.matmul(torch.transpose(s1, 1, 2), y)  # [B, k, C]
        adj1 = torch.matmul(torch.transpose(s1, 1, 2), adj)
        adj1 = torch.matmul(adj1, s1)  # [B, k, k]

        #adj1 = cutselfloop(adj1)
        #adj1 = normalize(adj1)
        #z = self.AttentionMP1(z)

        #print('adj1_grad', adj1.requires_grad)
        #print('y_grad', y.requires_grad)
        #z = self.NonLocalMP1(z)

        #z, F_score, vertices2 = self.View_selector1(y2,vertices,k=4)
        z, pooled_view2, s2 = self.GCN2(z, adj1)
        #pooled_view2 = torch.max(z, 1)[0]
        #pool_view2 = torch.mean(z, dim=1, keepdim=False)
        #z2 = self.NonLocalMP2(z)

        '''pooled_view2 = torch.max(z, 1)[0]
        pool_view2 = torch.mean(z, dim=1, keepdim=False)'''

        #s2 = self.AssignMatrix2(z, adj1)  # [B, k, k1]
        #print('adj1', adj1)
        #print('s2', s2)

        #mincut_loss2, ortho_loss2 = dense_mincut_pool(z, adj1, s2, mask=None)
        #link_loss2, ent_loss2 = dense_diff_pool(z, adj1, s2, mask=None)

        #loss = mincut_loss1 + ortho_loss1 + mincut_loss2 + ortho_loss2
        #loss = link_loss1 + link_loss2 + ent_loss1 + ent_loss2
        #loss = 5 * ortho_loss1 + 5 * ortho_loss2
        '''print('mincut1', mincut_loss1)
        print('mincut2', mincut_loss2)
        print('ortho1', ortho_loss1)
        print('ortho2', ortho_loss2)
        print('loss', loss)'''

        w = torch.matmul(torch.transpose(s2, 1, 2), z)  # [B, k1, C]
        adj2 = torch.matmul(torch.transpose(s2, 1, 2), adj1)
        adj2 = torch.matmul(adj2, s2)  # [B, k1, k1]
        #adj2 = cutselfloop(adj2)
        #adj2 = normalize(adj2)
        #w = self.AttentionMP2(w)
        #w = self.NonLocalMP2(w)
        #print('adj2', adj2)
        #w, F_score2, vertices3 = self.View_selector2(z2,vertices2,k=4)

        w, pooled_view3, s3 = self.GCN3(w, adj2)

        '''pooled_view3 = torch.max(w, 1)[0]
        pool_view3 = torch.mean(w, dim=1, keepdim=False)'''

        pooled_view = torch.cat((pooled_view1, pooled_view2, pooled_view3), 1)
        pooled_view4 = self.cls1(pooled_view)
        pooled_view5 = self.cls2(pooled_view4)
        #print('modeltime %.6f' % model_time.value())
        return pooled_view5, y, adj, s1, z, adj1, s2, pooled_view4
