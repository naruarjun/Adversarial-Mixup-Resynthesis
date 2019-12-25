import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import os
import pickle



class ModelHandler:
    def __init__(self, args, train_loader, val_loader, generator, discriminator, mixer, classifier):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_cuda = not args['no_cuda']
        if args['no_cuda']:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
        if args['dropout'] is not None:
            self.dropout = nn.Dropout2d(args['dropout'])
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        self.mixer = mixer
        self.gen_opt = Adam(self.generator.parameters(), lr = args['lr'])
        self.dis_opt = Adam(self.discriminator.parameters(), lr = args['lr'])

        self.lamb = args['lamb']
        self.beta = args['beta']
        self.n_classes = 10
        self.n_in = 32
        # self.fn = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(self.n_in, self.n_classes),
        #     nn.LogSoftmax(dim=1)
        # )
        self.fn = classifier
        self.epoch = 0
        self.fn.to(self.device)
        self.fn_opt = Adam(self.fn.parameters(), lr = args['lr'])
        if self.args['restore_dir']:
            self.restore()
        self.loss_train = {}
        self.loss_val = {}
        self.accuracy = {}

    def _run_epoch(self, dataloader, train = True):
        running_loss = 0
        abc = 0
        correct_all = 0
        total = 0
        for i, d in enumerate(dataloader):
            abc += 1
            update_g = False
            if i % self.args['update_g_freq']:
                update_g = True
            dec_mix, dec_loss, gen_loss, cls_loss, correct, no = self.train_on_instance(d, train, update_g)
            running_loss += cls_loss
            correct_all += correct
            total += no
            ########
            # Report Metric Here
            ########
        running_loss = running_loss/abc

        accuracy = correct_all/(total)
        if train:
            self.loss_train[self.epoch] = running_loss
            print("LOSS TRAIN:", running_loss,"  ACCURACY:",accuracy)
            
        else:
            self.accuracy[self.epoch] = accuracy
            self.loss_val[self.epoch] = running_loss
            print("LOSS VAL:", running_loss,"  ACCURACY:",accuracy)
            print(" ")
            


    def train(self):
        f1=open('mnist_train_loss_mixup2.p', 'a+')
        f2=open('mnist_val_loss_mixup2.p', 'a+')
        f3=open('mnist_acc_mixup2.p', 'a+')
        print(self.generator)
        print(self.discriminator)
        while self.epoch < self.args['epochs']:
            print("EPOCH: ",self.epoch)
            self._run_epoch(self.train_loader, train = True)
            ###
            # Report Metric
            ####

            self._run_epoch(self.val_loader, train = False)
            self.save_state()
            self.epoch += 1

            with open('pickle_data/mnist_train_loss_mixup2.p', 'wb') as handle:
                pickle.dump(self.loss_train,handle)
            with open('pickle_data/mnist_val_loss_mixup2.p', 'wb') as handle:
                pickle.dump(self.loss_val,handle)
            with open('pickle_data/mnist_acc_mixup2.p', 'wb') as handle:
                pickle.dump(self.accuracy,handle)

    

    def gan_loss(self, predictions, target):
        target = torch.ones_like(predictions) * target
        target = target.to(self.device)
        loss = nn.BCELoss()
        target = target.view(-1, 1)
        return loss(predictions, target)

    def accuracy(self,outputs,labels):
    	_, predicted = torch.max(outputs.data, 1)
    	correct += (predicted == labels).sum().item()
    	return correct


    def train_on_instance(self, x, train = True, update_g = False):
        labels = x[1]
        labels = labels.cuda()
        x = x[0]
        x = x.cuda()
        x_enc = self.generator.encode(x)
        if self.args['dropout'] is not None:
            x_enc = self.dropout(x_enc)
        x_enc_dec = self.generator.decode(x_enc)
        perm = torch.randperm(x.size(0))
        recon_loss = torch.mean(torch.abs(x_enc_dec - x))
        disc_loss_recon_g = self.gan_loss(self.discriminator(x_enc_dec), 1)
        is_2d = True if len(x_enc.size()) == 2 else False
        alpha = self.sampler(x.size(0), x_enc.size(1), is_2d)
        mix = alpha * x_enc + (1 - alpha) * x_enc[perm]
        dec_mix = self.generator.decode(mix)
        disc_loss_mix_g = self.gan_loss(self.discriminator(dec_mix), 1)
        gen_loss = recon_loss * self.lamb + disc_loss_recon_g + disc_loss_mix_g
        if train:
            if update_g:
                self.generator.zero_grad()
                gen_loss.backward()
                self.gen_opt.step()
        disc_loss_real = self.gan_loss(self.discriminator(x), 1)
        disc_loss_fake_recon = self.gan_loss(self.discriminator(x_enc_dec.detach()), 0)
        disc_loss_fake_mix = self.gan_loss(self.discriminator(dec_mix.detach()), 0)
        disc_loss = disc_loss_real + disc_loss_fake_mix + disc_loss_fake_recon
        if train:   
            self.discriminator.zero_grad()
            disc_loss.backward()
            self.dis_opt.step()
        outputs_cls = self.fn(x_enc.detach())
        criterion = nn.NLLLoss()
        cls_loss = criterion(outputs_cls, labels)
        _, predicted = torch.max(outputs_cls.data, 1)
        
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        if train:
            self.fn.zero_grad()
            cls_loss.backward()
            self.fn_opt.step()
        return dec_mix, disc_loss, gen_loss, cls_loss,correct,total

    def restore(self):
        state = torch.load(self.args['restore_dir'])
        self.epoch = state['epoch']
        self.generator.load_state_dict(state['generator'])
        self.discriminator.load_state_dict(state['discriminator'])
        self.fn.load_state_dict(state['classifier'])
        print(self.generator)
    def save_state(self):
        if not os.path.exists(self.args['save_state_dir']):
            os.mkdir(self.args['save_state_dir'])
        save_dic = {'epoch':self.epoch,
            'generator':self.generator.state_dict(),
            'discriminator':self.discriminator.state_dict(),
            'classifier':self.fn.state_dict()}
        torch.save(save_dic, self.args['save_state_dir']+'model_low_learning_rate.pth')

    def sampler_mixup(self, bs, f, is_2d, p=None):
        """Mixup sampling function
        :param bs: batch size
        :param f: number of features / channels
        :param is_2d: should sampled alpha be 2D, instead of 4D?
        :param p: Bernoulli parameter `p`. If this is None, then
          we simply sample p ~ U(0,1).
        :returns: an alpha of shape (bs, 1) if `is_2d`, otherwise
          (bs, 1, 1, 1).
        :rtype: 
        """
        shp = (bs, 1) if is_2d else (bs, 1, 1, 1)
        if p is None:
            alphas = []
            for i in range(bs):
                alpha = np.random.uniform(0, 1)
                alphas.append(alpha)
        else:
            alphas = [p]*bs
        alphas = np.asarray(alphas).reshape(shp)
        alphas = torch.from_numpy(alphas).float()
        if self.use_cuda:
            alphas = alphas.cuda()
        return alphas

    def sampler_mixup2(self, bs, f, is_2d, p=None):
        """Mixup2 sampling function
        :param bs: batch size
        :param f: number of features / channels
        :param is_2d: should sampled alpha be 2D, instead of 4D?
        :param p: Bernoulli parameter `p`. If this is None, then
          we simply sample p ~ U(0,1).
        :returns: an alpha of shape (bs, f) if `is_2d`, otherwise
          (bs, f, 1, 1).
        :rtype: 
        """
        shp = (bs, f) if is_2d else (bs, f, 1, 1)
        if p is None:
            alphas = np.random.uniform(0, 1, size=shp)
        else:
            alphas = np.zeros(shp)+p
        alphas = torch.from_numpy(alphas).float()
        if self.use_cuda:
            alphas = alphas.cuda()
        return alphas

    def sampler_fm(self, bs, f, is_2d, p=None):
        """Bernoulli mixup sampling function
        :param bs: batch size
        :param f: number of features / channels
        :param is_2d: should sampled alpha be 2D, instead of 4D?
        :param p: Bernoulli parameter `p`. If this is `None`, then
          we simply sample m ~ Bern(p), where p ~ U(0,1).
        :returns: an alpha of shape (bs, f) if `is_2d`, otherwise
          (bs, f, 1, 1).
        :rtype: 
        """
        shp = (bs, f) if is_2d else (bs, f, 1, 1)
        if p is None:
            alphas = torch.bernoulli(torch.rand(shp)).float()
        else:
            rnd_state = np.random.RandomState(0)
            rnd_idxs = np.arange(0, f)
            rnd_state.shuffle(rnd_idxs)
            rnd_idxs = torch.from_numpy(rnd_idxs)
            how_many = int(p*f)
            alphas = torch.zeros(shp).float()
            if how_many > 0:
                rnd_idxs = rnd_idxs[0:how_many]
                alphas[:, rnd_idxs] += 1.
        if self.use_cuda:
            alphas = alphas.cuda()
        return alphas

    def sampler_fm2(self, bs, f, is_2d, p=None):
        """Bernoulli mixup sampling function. Has
          same expectation as fm but higher variance.
        :param bs: batch size
        :param f: number of features / channels
        :param is_2d: should sampled alpha be 2D, instead of 4D?
        :param p: Bernoulli parameter `p`. If this is `None`, then
          we simply sample m ~ Bern(p), where p ~ U(0,1).
        :returns: an alpha of shape (bs, f) if `is_2d`, otherwise
          (bs, f, 1, 1).
        :rtype: 
        """
        shp = (bs, f) if is_2d else (bs, f, 1, 1)
        if p is None:
            this_p = torch.rand(1).item()
            alphas = torch.bernoulli(torch.zeros(shp)+this_p).float()
        else:
            rnd_state = np.random.RandomState(0)
            rnd_idxs = np.arange(0, f)
            rnd_state.shuffle(rnd_idxs)
            rnd_idxs = torch.from_numpy(rnd_idxs)
            how_many = int(p*f)
            alphas = torch.zeros(shp).float()
            if how_many > 0:
                rnd_idxs = rnd_idxs[0:how_many]
                alphas[:, rnd_idxs] += 1.
        if self.use_cuda:
            alphas = alphas.cuda()
        return alphas


    def sampler(self, bs, f, is_2d, **kwargs):
        """Sampler function, which outputs an alpha which
        you can use to produce a convex combination between
        two examples.
        :param bs: batch size
        :param f: number of units / feature maps at encoding
        :param is_2d: is the bottleneck a 2d tensor?
        :returns: an alpha of shape `(bs, f)` is `is_2d` is set,
          otherwise `(bs, f, 1, 1)`.
        :rtype: 
        """
        if self.mixer == 'mixup':
            return self.sampler_mixup(bs, f, is_2d, **kwargs)
        elif self.mixer == 'mixup2':
            return self.sampler_mixup2(bs, f, is_2d, **kwargs)
        elif self.mixer == 'fm':
            return self.sampler_fm(bs, f, is_2d, **kwargs)
        elif self.mixer == 'fm2':
            return self.sampler_fm2(bs, f, is_2d, **kwargs)
