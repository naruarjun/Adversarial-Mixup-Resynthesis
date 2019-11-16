import torch
import torch.nn as nn
from torch.optim import Adam
class ModelHandler:
    def __init__(self, args, train_loader, val_loader, generator, discriminator, mixer):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
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

    def _run_epoch(self, dataloader, train = True):
        for i, d in enumerate(dataloader):
            update_g = False
            if i % self.args['update_g_freq']:
                update_g = True
            dec_mix, dec_loss, gen_loss = self.train_on_instance(d, train, update_g)

            ########
            # Report Metric Here
            ########


    def train(self):
        for i in range(self.args['epochs']):
            self._run_epoch(self.train_loader, train = True)

            ###
            # Report Metric
            ####

            self._run_epoch(self.val_loader, train = False)

    

    def gan_loss(self, predictions, target):
        target = torch.ones_like(predictions) * target
        target = target.to(self.device)
        loss = nn.BCELoss()
        target = target.view(-1, 1)
        return loss(predictions, target)


    def train_on_instance(self, x, train = True, update_g = False):
        x_enc = self.generator.encode(x)
        if args['dropout'] is not None:
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
        gen_loss = self.recon_loss * self.lamb + disc_loss_recon_g + disc_loss_mix_g
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
        return dec_mix, disc_loss, gen_loss




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
