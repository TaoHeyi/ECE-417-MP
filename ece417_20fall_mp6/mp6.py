# coding: utf-8

import os, argparse, glob, time, math

import tqdm, skvideo.io, numpy as np

import torch
from torch import empty
from torch import zeros_like
from torch import matmul
from torch import stack, zeros, ones, manual_seed, load, randn, cat
from torch.autograd import Variable
from torch.nn import Module, Parameter, Sequential, ConvTranspose2d, BatchNorm2d, ReLU, Tanh, BatchNorm1d, Sigmoid, Linear
from torch.nn.init import constant_, xavier_uniform_

################################################################################

class MyLinear(Module):
    def __init__(self, input_size, output_size):
        """ TODO:
        Sets up the following Parameters:
            self.weight - A Parameter holding the weights of the layer, of size (output_size, input_size).
            self.bias - A Parameter holding the biases of the layer, of size (output_size,).
        Note that these have the same names as their equivalents in torch.nn.LSTMCell
        and thus serve similar purposes. Feel free to initialize these how you like here; prior to their use,
        a separate method will initialize these (which you will also need to fill in below).
        You may also set other diagnostics at this point, but these are not strictly necessary."""
        super(MyLinear, self).__init__()
        #raise NotImplementedError("You need to write this part!")
        self.input_size = input_size
        self.output_size = output_size
        self.weight = randn(output_size, input_size)
        self.bias = randn(output_size)

        #self.linear1 = randn(input_size, output_size)
        #self.relu1 = ReLU()
        #self.linear2 = Linear(35, output_size)
        #self.relu2 = ReLU()
        #self.linear3 = Linear(100, output_size)
        #self.relu3 = ReLU()
    
    def forward(self, inputs):
        """ TODO:
        Performs the forward propagation of a linear layer.
        Input:
            inputs - the input to the cell, of size (input_size,)
        Output:
            (your name here) - the output to the cell, of size (output_size,) """
        #raise NotImplementedError("You need to write this part!")
        #print(inputs)
        return matmul(inputs, self.weight.t()) + self.bias

class MyLSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        """ TODO:
        # Sets up the following Parameters:
        #     self.weight_ih - A Parameter holding the weights w_i, w_f, w_c, and w_o,
        #         each of size (4*hidden_size, input_size), concatenated along the first dimension
        #         so that the total size of the tensor is (4*hidden_size, input_size).
        #     self.bias_ih - A Parameter holding the components of the biases b_i, b_f, b_c, and b_o
        #         with respect to the parts of self.weight_ih, each of size (4*hidden_size,) and similarly
        #         concatenated along the first dimension to yield a (4*hidden_size,) tensor.
        #     self.weight_hh - A Parameter holding the weights u_i, u_f, u_c, and u_o,
        #         of the same shape and structure as self.weight_ih.
        #     self.bias_hh - A Parameter holding the components of the biases b_i, b_f, b_c, and b_o
        #         with respect to the parts of self.weight_hh, of the same shape and structure as self.bias_ih.
        # Note that these have the same names as their equivalents in torch.nn.LSTMCell
        # and thus serve similar purposes. Feel free to initialize these how you like here; prior to their use,
        # a separate method will initialize these (which you will also need to fill in below).
        # You may define layers for the Tanh and Sigmoid functions here, and you may also save other
        # diagnostics at this point, but these are not strictly necessary. """
        super(MyLSTMCell, self).__init__()
        #print("Input size is ", input_size) #"9"
        #print("Hidden size is ", hidden_size) #"15"
        #raise NotImplementedError("You need to write this part!")

        #self.input_size = input_size
        #self.hidden_size = hidden_size

        self.weight_ih = Parameter(randn(4*hidden_size, input_size))
        self.bias_ih = Parameter(randn(4*hidden_size))
        self.weight_hh = Parameter(randn(4*hidden_size, hidden_size))
        self.bias_hh = Parameter(randn(4*hidden_size))
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()
    
    def forward(self, x, h_in_c_in):
        """ TODO:
        Performs the forward propagation of an LSTM cell.
        Inputs:
           x - The input to the cell, of size (batch_size,input_size)
           h_in_c_in - A tuple (h_in, c_in) consisting of the following:
             h_in - The initial hidden state h for this timestep, of size (batch_size,hidden_size).
             c_in - The initial memory cell c for this timestep, of size (batch_size,hidden_size).
        Outputs:
           h_out - The resulting (hidden state) output h, of the same size as h_in.
           c_out - The resulting memory cell c, of the same size as c_in.
        Note that c_out should be passed through a tanh layer before it is used when
        computing h_out. """
        h_in = h_in_c_in[0]
        c_in = h_in_c_in[1]
        #print(h_in)
        #print(self.weight_ih.t())
        #print(c_in)

        result1 = torch.mm(x, self.weight_ih.t()) + self.bias_ih
        result2 = torch.mm(h_in, self.weight_hh.t()) + self.bias_hh
        gates = (result1 + result2)
        gates = gates.chunk(4,1)
        #print(final)
        #print(gates)

        gates_i = gates[0]
        gates_f = gates[1]
        gates_c = gates[2]
        gates_o = gates[3]
        #print(gates_i)

        gates_i = torch.sigmoid(gates_i)
        gates_f = torch.sigmoid(gates_f)
        gates_c = torch.tanh(gates_c)
        gates_o = torch.sigmoid(gates_o)

        c_out = gates_f*c_in + gates_i*gates_c
        h_out = gates_o*torch.tanh(c_out)

        return h_out, c_out


class LSTM(Module):
    def __init__(self, input_size, hidden_size):
        """ TODO:
        Sets up the following:
            self.lstm - a MyLSTMCell with input_size and hidden_size as constructor inputs.
            self.linear - a MyLinear layer with hidden_size and input_size as constructor inputs.
            self.bn - a BatchNorm1d layer of size input_size which does NOT have learnable affine parameters.
            self.hidden_size - the hidden size passed in to this constructor.
        You may also save the input size if that becomes useful to you later. """
        super(LSTM, self).__init__()
        #raise NotImplementedError("You need to write this part!")
        self.lstm = MyLSTMCell(input_size, hidden_size)
        self.linear = MyLinear(hidden_size, input_size)
        self.bn = BatchNorm1d(input_size, affine=False)
        self.hidden_size = hidden_size

    def initWeight(self, init_forget_bias=1):
        """ TODO:
        Goes through the parameters of this Module and operates on the weights
        as follows:
            - Any 'weight's should be initialized using the xavier_uniform_ distribution.
            - The 'bias'es used with the forget gate should be set to init_forget_bias.
              All other biases should be zero.
              (Here you might find torch.Tensor.chunk useful.)
            - All other parameters can be set to zero. """
        #raise NotImplementedError("You need to write this part!")
        xavier_uniform_(self.lstm.weight_ih)
        xavier_uniform_(self.lstm.weight_hh)
        xavier_uniform_(self.linear.weight)

        temp_bias_ih = zeros_like(self.lstm.bias_ih)
        temp_bias_ih[1*self.hidden_size : 2*self.hidden_size] += init_forget_bias
        self.lstm.bias_ih = Parameter(temp_bias_ih)

        temp_bias_hh = zeros_like(self.lstm.bias_hh)
        temp_bias_hh[1 * self.hidden_size: 2 * self.hidden_size] += init_forget_bias
        self.lstm.bias_hh = Parameter(temp_bias_hh)

        self.linear.bias = Parameter(zeros_like(self.linear.bias))

    def initHidden(self, batch_size):
        """ TODO:
        Defines an initial hidden state self.h and an initial memory cell self.c,
        both of size (batch_size, self.hidden_size), as Variables containing all zeros. """
        #raise NotImplementedError("You need to write this part!")
        self.h = zeros(batch_size, self.hidden_size)
        self.c = zeros(batch_size, self.hidden_size)

    def forward(self, inputs, n_frames):
        """ TODO:
        Produces vectors in the motion subspace of the latent space of images
        (from which the generator generates video frames).
        Assume that initHidden has already been run. Iterate for n_frames,
        repeating the same inputs as the input to every frame in the sequence.
        After each frame, the hidden state output should be saved,
        then each of those saved states should be passed through the linear and
        batch normalization layers, and the results of these should be stacked
        along a new first dimension.
        
        Inputs:
            inputs - The inputs to the LSTM cell, of size (batch_size, input_size).
        Outputs:
            outputs - Stacked hidden states, of size (n_frames, batch_size, input_size). """
        #raise NotImplementedError("You need to write this part!")

        d1 = inputs.shape[0]
        d2 = inputs.shape[1]
        outputs = empty(n_frames, d1, d2)
        for i in range(n_frames):
            self.h, self.c = self.lstm(inputs, (self.h, self.c))
            outputs[i] = self.bn(self.linear(self.h))
        return outputs

################################################################################

if(__name__ == "__main__"):
    img_size = 96
    nc = 3
    ndf = 64
    ngf = 64
    d_E = 10
    hidden_size = 100 # arbitrary
    d_C = 50
    d_M = d_E
    nz  = d_C + d_M

    class Generator_I(Module):
        def __init__(self, nc=3, ngf=64, nz=60, ngpu=1):
            super(Generator_I, self).__init__()
            self.ngpu = ngpu
            self.main = Sequential(
                ConvTranspose2d(     nz, ngf * 8, 6, 1, 0, bias=False),
                BatchNorm2d(ngf * 8),
                ReLU(True),
                ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                BatchNorm2d(ngf * 4),
                ReLU(True),
                ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                BatchNorm2d(ngf * 2),
                ReLU(True),
                ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
                BatchNorm2d(ngf),
                ReLU(True),
                ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
                Tanh()
            )

        def forward(self, input):
            return self.main(input)

    parser = argparse.ArgumentParser(description='Start using MoCoGAN')
    parser.add_argument('--batch-size', type=int, default=16, help='set batch_size, default: 16')
    parser.add_argument('--niter', type=int, default=16, help='set num of iterations, default: 120000')

    args       = parser.parse_args()
    batch_size = args.batch_size
    n_iter     = args.niter

    seed = 2020
    manual_seed(seed)
    np.random.seed(seed)

    current_path = os.getcwd()

    # Lengths of videos in the dataset "Actions as Space-Time Shapes".
    video_lengths = [84, 85, 55, 51, 54, 70, 146, 56, 105, 103, 67, 67, 63, 45, 43, 72, 47, 40, 39, 38, 62, 54, 42, 65, 48, 127, 49, 55, 45, 56, 42, 41, 76, 36, 93, 56, 52, 56, 36, 64, 67, 53, 52, 48, 39, 64, 64, 60, 39, 46, 43, 57, 48, 60, 38, 39, 63, 92, 85, 37, 77, 68, 84, 68, 101, 43, 88, 61, 119, 112, 50, 111, 120, 82, 60, 125, 55, 103, 61, 53, 54, 60, 61, 81, 51, 54, 67, 114, 79, 89, 57, 62, 59]
    n_videos = len(video_lengths)
    T = 16

    def trim_noise(noise):
        start = np.random.randint(0, noise.size(1) - (T+1))
        end = start + T
        return noise[:, start:end, :, :, :]


    gen_i = Generator_I(nc, ngf, nz).requires_grad_(False)
    lstm = LSTM(d_E, hidden_size).requires_grad_(False)
    lstm.initWeight()

    trained_path = os.path.join(current_path, 'data')

    def save_video(fake_video, index):
        outputdata = fake_video * 255
        outputdata = outputdata.astype(np.uint8)
        dir_path = os.path.join(current_path, 'outputs')
        file_path = os.path.join(dir_path, 'video_%d.mp4' % index)
        skvideo.io.vwrite(file_path, outputdata)

    gen_i.load_state_dict(load(trained_path + '\Generator_I-120000.model'))
    lstm.load_state_dict(load(trained_path + '\LSTM-120000.model'))

    ''' generate motion and content latent space vectors '''
    def gen_z(n_frames):
        z_C = Variable(randn(batch_size, d_C))
        z_C = z_C.unsqueeze(1).repeat(1, n_frames, 1)
        eps = Variable(randn(batch_size, d_E))

        lstm.initHidden(batch_size)
        z_M = lstm(eps, n_frames).transpose(1, 0)
        z = cat((z_M, z_C), 2)
        return z.view(batch_size, n_frames, nz, 1, 1)

    for epoch in tqdm.tqdm(range(n_iter)):
        # note that n_frames is sampled from video length distribution
        n_frames = video_lengths[np.random.randint(0, len(video_lengths))]
        Z = gen_z(n_frames)  # Z.size() => (batch_size, n_frames, nz, 1, 1)
        # trim => (batch_size, T, nz, 1, 1)
        Z = trim_noise(Z)
        # generate videos
        Z = Z.contiguous().view(batch_size*T, nz, 1, 1)
        fake_videos = gen_i(Z)
        fake_videos = fake_videos.view(batch_size, T, nc, img_size, img_size)
        fake_videos = fake_videos.transpose(2, 1)
        fake_img = fake_videos[:, :, np.random.randint(0, T), :, :]

    for iii in range(fake_videos.shape[0]):
        save_video(fake_videos[iii].data.cpu().numpy().transpose(1, 2, 3, 0), iii)
