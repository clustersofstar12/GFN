import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from tools import meter
from ptflops import get_model_complexity_info
from tools.diffpool import dense_diff_pool

class ModelNetTrainer(object):
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, \
                 model_name, log_dir, num_views=12):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views
        self.model.cuda()
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)
    def train(self, n_epochs):
        best_acc = 0
        i_acc = 0

        #batch_time = meter.TimeMeter(True)
        #epoch_time = meter.TimeMeter(True)

        '''if self.model_name == 'view-gfn':
            total_num = sum(p.numel() for p in self.model.parameters())
            trainable_num = sum(q.numel() for q in self.model.parameters() if q.requires_grad)
            print('total', total_num)
            print('trainable', trainable_num)'''

        '''input = torch.randn(20, 3, 224, 224).cuda()
        flops, params = profile(self.model, inputs=(input, ))
        print('flops', flops)
        print('params', params)'''

        '''macs, params = get_model_complexity_info(self.model, (20, 512), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
        print('flops', macs)
        print('param', params)'''

        self.model.train()
        scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.8, patience=5, verbose=True)
        for epoch in range(n_epochs):
            #epoch_time.reset()
            total_num = sum(p.numel() for p in self.model.parameters())
            trainable_num = sum(q.numel() for q in self.model.parameters() if q.requires_grad)
            print('total', total_num)
            print('trainable', trainable_num)

            if self.model_name == 'view-gfn':
                '''if epoch == 20:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = 1e-4'''

                if (epoch + 1) % 15 == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.8

                '''if epoch > 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5 * (1 + math.cos(epoch * math.pi / 15))'''

            else:
                if epoch > 0 and (epoch + 1) % 10 == 0:
                    for param_group in self.optimizer.param_groups:
                        #param_group['lr'] = param_group['lr'] * 0.7
                        param_group['lr'] = param_group['lr']

            # permute data for mvcnn
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths) / self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[
                                     rand_idx[i] * self.num_views:(rand_idx[i] + 1) * self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new
            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print('lr', lr)
            self.writer.add_scalar('params/lr', lr, epoch)
            # train one epoch
            out_data = None
            in_data = None

            compution_time = 0
            epoch_time = time.time()
            for i, data in enumerate(self.train_loader):
                #batch_time.reset()
                if self.model_name == 'view-gfn' and epoch == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr * ((i + 1) / (len(rand_idx) // 20))
                if self.model_name == 'view-gfn':
                    N, V, C, H, W = data[1].size()
                    in_data = Variable(data[1]).view(-1, C, H, W).cuda()
                else:
                    in_data = Variable(data[1].cuda())
                    #print(in_data.size())
                target = Variable(data[0]).cuda().long()
                target_ = target.unsqueeze(1).repeat(1, 4*(10+5)).view(-1)
                self.optimizer.zero_grad()

                batch_time = time.time()
                if self.model_name == 'view-gfn':
                    out_data, y, adj, s1, z, adj1, s2, ft = self.model(in_data)
                    #print('batchtime %.6f' % batch_time.value())
                    '''if (epoch + 1) % 10 == 5:
                        print('adj1', adj1)
                        print('s1', s1)
                        print('adj2', adj2)
                        print('s2', s2)'''
                    #out_data_ = torch.cat((F_score, F_score2), 1).view(-1, 40)
                    link_loss1, ent_loss1 = dense_diff_pool(y, adj, s1, mask=None)
                    link_loss2, ent_loss2 = dense_diff_pool(z, adj1, s2, mask=None)
                    loss1 = link_loss1 + link_loss2 + ent_loss1 + ent_loss2
                    loss = self.loss_fn(out_data, target) + loss1
                    #loss = self.loss_fn(out_data, target)
                else:
                    out_data = self.model(in_data)
                    loss = self.loss_fn(out_data, target)

                #self.writer.add_scalar('train/train_loss', loss, i_acc + i + 1)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float() / results.size()[0]
                #self.writer.add_scalar('train/train_overall_acc', acc, i_acc + i + 1)
                #print('lr = ', str(param_group['lr']))
                loss.backward()
                if self.model_name == 'view-gfn':
                    clip_grad_norm_(self.model.parameters(), max_norm=3, norm_type=2)

                '''if self.model_name == 'view-gfn':
                    for name, parms in self.model.named_parameters():
                        print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:',parms.grad)'''
                self.optimizer.step()
                print('batchtime %.6f' % (time.time()-batch_time))
                compution_time += time.time() - batch_time

                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch + 1, i + 1, loss, acc)
                if (i + 1) % 1 == 0:
                    print(log_str)
            #i_acc += i

            print('compution_time %.6f' % compution_time)
            print('epochtime %.6f' % (time.time() - epoch_time))

            # evaluation
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    test_loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch)

                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                self.writer.add_scalar('val/val_loss', test_loss, epoch + 1)
                self.model.save(self.log_dir, epoch)
            # save best model
                if val_overall_acc > best_acc:
                    best_acc = val_overall_acc
                print('best_acc', best_acc)
            scheduler.step(test_loss)
        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir + "/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0
        count = 0
        wrong_class = np.zeros(51)
        samples_class = np.zeros(51)
        all_loss = 0
        #retrieval_map = meter.RetrievalMAPMeter()
        best_map = 0
        self.model.eval()

        for _, data in enumerate(self.val_loader, 0):

            if self.model_name == 'view-gfn':
                N, V, C, H, W = data[1].size()
                in_data = Variable(data[1]).view(-1, C, H, W).cuda()
            else:  # 'svcnn'
                in_data = Variable(data[1]).cuda()
            target = Variable(data[0]).cuda()
            if self.model_name == 'view-gfn':
                out_data, y, adj, s1, z, adj1, s2, ft = self.model(in_data)
                link_loss1, ent_loss1 = dense_diff_pool(y, adj, s1, mask=None)
                link_loss2, ent_loss2 = dense_diff_pool(z, adj1, s2, mask=None)
                loss1 = link_loss1 + link_loss2 + ent_loss1 + ent_loss2
                all_loss = all_loss + self.loss_fn(out_data, target).cpu().data.numpy() + loss1.cpu().data.numpy()
                #retrieval_map.add(ft.detach() / torch.norm(ft.detach(), 2, 1, True), target.detach())
            else:
                out_data = self.model(in_data)
                all_loss = all_loss + self.loss_fn(out_data, target).cpu().data.numpy()
            pred = torch.max(out_data, 1)[1]
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print('Total # of test models: ', all_points)
        class_acc = (samples_class - wrong_class) / samples_class
        val_mean_class_acc = np.mean(class_acc)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)
        '''if self.model_name == 'view-gfn':
            map = retrieval_map.mAP()
            if map > best_map:
                best_map = map
            print('map', map)'''
        print('val mean class acc. : ', val_mean_class_acc)
        print('val overall acc. : ', val_overall_acc)
        print('val loss : ', loss)
        print(class_acc)
        #print('best_map', best_map)
        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc
