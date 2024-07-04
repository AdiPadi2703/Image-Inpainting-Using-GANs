# This file contains the layers and blocks used in the models for the Generator and Discriminator.
# Sources and references are mentioned along with each class definition


import torch
from torch.autograd import Variable
from torch import nn

#########################################################################################################

'''

Implementation of Spectral Normalization for PyTorch.

Original Source: https://gist.github.com/rosinality/a96c559d84ef2b138e486acf27b5a56e

PyTorch has a built in function for spectral norm, however it is not compatible with torch.cuda.amp (mixed precision)
since torch.dot is not added to the list of operations that are compatible with amp.

'''

class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        if weight_mat.is_cuda:
            u = u.cuda()
        v = weight_mat.t() @ u
        v = v / v.norm()
        u = weight_mat @ v
        u = u / u.norm()
        weight_sn = weight_mat / (u.t() @ weight_mat @ v)
        weight_sn = weight_sn.view(*size)

        return weight_sn, Variable(u.data)

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        input_size = weight.size(0)
        u = Variable(torch.randn(input_size, 1) * 0.1, requires_grad=False)
        setattr(module, name + '_u', u)
        setattr(module, name, fn.compute_weight(module)[0])

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)

def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

    return module


#######################################################################################################33


class GatedConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="auto", rate=1, gated=True):

        super(GatedConv, self).__init__()
        padding = rate*(kernel_size-1)//2 if padding == "auto" else padding
        self.gated =gated
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding, dilation=rate,bias=False)


    def forward(self, x):

        if not self.gated:
            x = self.conv(x)
            return x
        else:
            x_ = self.conv(x)
            y = self.conv(x)
            x_ = torch.nn.functional.elu(x_)
            y = torch.sigmoid(y)
            out = x_ * y
            return out

############################################################################################################


class GatedDeconv(nn.Module):

    def __init__(self, in_channels, out_channels, padding=1):

        super(GatedDeconv, self).__init__()

        self.conv = GatedConv(in_channels, out_channels, kernel_size = 3, stride = 1, padding = padding)

    def forward(self, x):

        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest", recompute_scale_factor=2)
        x = self.conv(x)

        return x


###############################################################################################################


class DownSample(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None):

        super(DownSample, self).__init__()

        hidden_channels = out_channels if hidden_channels == None else hidden_channels
        self.conv1 = GatedConv(in_channels, hidden_channels, kernel_size=3, stride=2)
        self.conv2 = GatedConv(hidden_channels, out_channels, kernel_size=3, stride=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x

##############################################################################################################

class UpSample(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=None):

        super(UpSample, self).__init__()

        hidden_channels = out_channels if hidden_channels == None else hidden_channels
        self.conv1 = GatedDeconv(in_channels, hidden_channels)
        self.conv2 = GatedConv(hidden_channels, out_channels, kernel_size=3, stride=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x

##############################################################################################################

class SelfAttention(nn.Module):

    '''
    Self Attention for GANs : https://arxiv.org/abs/1805.08318
    '''

    def __init__(self,in_channels,activation):
        super(SelfAttention,self).__init__()

        self.activation = activation
        self.query = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//8 , kernel_size= 1)
        self.key = nn.Conv2d(in_channels = in_channels , out_channels = in_channels//8 , kernel_size= 1)
        self.value = nn.Conv2d(in_channels = in_channels , out_channels = in_channels , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) 
    def forward(self,x):
        
        m_batchsize,C,width ,height = x.size()
        query = self.query(x).view(m_batchsize,-1,width*height) 
        key = self.key(x).view(m_batchsize,-1,width*height) 
        s =  torch.bmm(query.permute(0,2,1),key) 
        beta = self.softmax(s) 
        value = self.value(x).view(m_batchsize,-1,width*height) 

        out = torch.bmm(value,beta.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out
        
#################################################################################################################

class ConvSN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding="auto"):

        super(ConvSN, self).__init__()
        padding = (kernel_size-1) //2 if padding == 'auto' else padding

        self.snconv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.leaky = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.snconv(x)
        x = self.leaky(x)
        return x

##########################################################################################################



    
    
        









































