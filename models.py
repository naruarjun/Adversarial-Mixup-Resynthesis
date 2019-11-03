import numpy
import torch
from torch import nn
import math
import torch.nn.functional as F


def Initialize(self,init,layers,slope=0.2):
	if(init=="xavier"):
		for layer in layers:
			if hasattr(layer, 'weight'):
				w = layer.weight.data
				nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
			if hasattr(layer, 'bias'):
				b = layer.bias.data
				nn.init.xavier_uniform_(b, gain=nn.init.calculate_gain('relu'))
	if(init=="dirac"):
		for layer in layers:
			if hasattr(layer, 'weight'):
				w = layer.weight.data
				nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
			if hasattr(layer, 'bias'):
				b = layer.bias.data
				nn.init.kaiming_uniform_(b, mode='fan_in', nonlinearity='relu')
	if(init=="kaiming"):
		for layer in layers:
			if hasattr(layer, 'weight'):
				w = layer.weight.data
				nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
			if hasattr(layer, 'bias'):
				b = layer.bias.data
				nn.init.kaiming_normal_(b, mode='fan_out', nonlinearity='relu')
	if(init=="uniform"):
		for layer in layers:
			if hasattr(layer, 'weight'):
				w = layer.weight.data
				nn.init.uniform_(w)
			if hasattr(layer, 'bias'):
				b = layer.bias.data
				nn.init.uniform_(b)
	if(init=="normal"):
		for layer in layers:
			if hasattr(layer, 'weight'):
				w = layer.weight.data
				nn.init.normal_(w)
			if hasattr(layer, 'bias'):
				b = layer.bias.data
				nn.init.normal_(b)
	if(init=="default"):
		for layer in layers:
			if hasattr(layer, 'weight'):
				w = layer.weight.data
				std = 1/np.sqrt((1 + slope**2) * np.prod(w.shape[:-1]))
				w.normal_(std=std)  
			if hasattr(layer, 'bias'):
				layer.bias.data.zero_()



class Encoder(nn.module):
	def __init__(self,scales,initial_depth,final_depth,depth,kernel_size=3,padding=1,instance=False,spectral=False,dropout=None,init='xavier'):
		super(Encoder,self).__init__()
		
		self.layers = []

		if spectral:
			sn = nn.utils.spectral_norm
		else:
			sn = lambda x:x

		self.layers.append(nn.Conv2d(depth,initial_depth,kernel_size,padding=padding))

		initial_depth_temp = initial_depth
		for scale in range(scales+1):
			new_depth = initial_depth << scale

			if instance:
				self.layers.append(nn.Conv2d(initial_depth_temp,new_depth,kernel_size,padding=padding))
				self.layers.append(nn.InstanceNorm2d(new_depth,affine=True))
				self.layers.append(nn.LeakyReLU())
			else:
				self.layers.append(nn.Conv2d(initial_depth_temp,new_depth,kernel_size,padding=padding))
				self.layers.append(nn.LeakyReLU())
			
			self.layers.append(nn.AvgPool2d(2))
			
			initial_depth_temp = new_depth

		self.layers.append(nn.Conv2d(initial_depth_temp,final_depth,kernel_size,padding=padding))
		
		if dropout:
			self.layers.append(nn.Dropout2d(dropout))
		
		Initialize(init,self.layers)
		
		self.model = nn.Sequential(*(self.layers))

	def forward(self,x):
		out = self.model(x);
		return out



class Decoder(nn.module):
	def __init__(self,scales,initial_depth,final_depth,depth,kernel_size=3,padding=1,instance=False,spectral=False,dropout=None,init='xavier'):
		uper(Encoder,self).__init__()

		self.layers = []
		
		if spectral:
			sn = nn.utils.spectral_norm
		else:
			sn = lambda x:x

		final_depth_temp = final_depth
			
		for scale in range(scales-1,-1,-1):
			new_depth = initial_depth << scale

			if instance:
				self.layers.append(nn.Conv2d(final_depth_temp,new_depth,kernel_size,padding=padding))
				self.layers.append(nn.InstanceNorm2d(new_depth,affine=True))
				self.layers.append(nn.LeakyReLU())
			else:
				self.layers.append(nn.Conv2d(final_depth_temp,new_depth,kernel_size,padding=padding))
				self.layers.append(nn.LeakyReLU())

			self.layers.append(nn.Upsample(scale_factor = 2))

			final_depth_temp = new_depth

		if instance_norm:
			self.layers.append(nn.Conv2d(final_depth_temp,initial_depth,kernel_size,padding=padding))
			self.layers.append(nn.InstanceNorm2d(initial_depth,affine=True))
			self.layers.append(nn.LeakyReLU())
		else:
			self.layers.append(nn.Conv2d(final_depth_temp,initial_depth,kernel_size,padding=padding))
			self.layers.append(nn.LeakyReLU())

		self.layers.append(nn.Conv2d(initial_depth, depth, kernel_size,padding=padding))

		Initialize(init,self.layers)

		self.model = nn.Sequential(*(self.layers))

    def forward(self,x):
		out = self.model(x);
		return out



class Autoencoder(nn.module):
	def __init__(self,scales,n_channels,initial_depth,final_depth,kernel_size=3,padding=1,instance=False,spectral=False,dropout=None,init='xavier'):
		super(Autoencoder,self).__init()

		self.encoder = Encoder(scales,initial_depth,final_depth,n_channels,instance=instance)
		self.decoder = Decoder(scales,initial_depth,final_depth,n_channels,instance=instance)

	def forward(self,x):
		enc = self.encoder(x)
		dec = self.decoder(enc)
		return dec

	def encode(self, x):
		return self.encoder(x)
	
	def decode(self, x):
		return self.decoder(x)



class Discriminator(nn.module):
	def __init__(self,scales,initial_depth,final_depth,depth,kernel_size=3,padding=1,instance=False,spectral=False,dropout=None,init='xavier')
		super(Discriminator,self).__init__()

		self.encoder = Encoder(scales,initial_depth,final_depth,depth,spectral=spectral)
		self.fc = nn.Linear(final_depth*(final_depth**2),2)

	def forward(self,x):
		x = self.encoder(x)
		x = x.reshape(x.shape[0],-1)
		x = self.fc(x)
		return F.log_softmax(x)






	








