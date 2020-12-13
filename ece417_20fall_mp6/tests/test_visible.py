import unittest, mp6, json
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
from torch import count_nonzero, Tensor, sum as torchsum, abs as torchabs
from torch.nn import BatchNorm1d, Parameter
from torch.autograd import Variable

class TestStep(unittest.TestCase):
    def setUp(self):
        with open('solutions.json') as f:
            data = json.load(f)
            self.linear_bias = Tensor(data["linear_bias"])
            self.linear_input = Tensor(data["linear_input"])
            self.linear_input_size = data["linear_input_size"]
            self.linear_output = Tensor(data["linear_output"])
            self.linear_output_size = data["linear_output_size"]
            self.linear_weight = Tensor(data["linear_weight"])
            self.lstm_batch_size = data["lstm_batch_size"]
            self.lstm_c = Tensor(data["lstm_c"])
            self.lstm_forget_bias = data["lstm_forget_bias"]
            self.lstm_h = Tensor(data["lstm_h"])
            self.lstm_inputs = Tensor(data["lstm_inputs"])
            self.lstm_linear_bias = Tensor(data["lstm_linear_bias"])
            self.lstm_linear_weight = Tensor(data["lstm_linear_weight"])
            self.lstm_n_frames = data["lstm_n_frames"]
            self.lstm_output = Tensor(data["lstm_output"])
            self.lstmcell_bias_hh = Tensor(data["lstmcell_bias_hh"])
            self.lstmcell_bias_ih = Tensor(data["lstmcell_bias_ih"])
            self.lstmcell_c_init = Tensor(data["lstmcell_c_init"])
            self.lstmcell_cout = Tensor(data["lstmcell_cout"])
            self.lstmcell_h_init = Tensor(data["lstmcell_h_init"])
            self.lstmcell_hidden_size = data["lstmcell_hidden_size"]
            self.lstmcell_input = Tensor(data["lstmcell_input"])
            self.lstmcell_input_size = data["lstmcell_input_size"]
            self.lstmcell_output = Tensor(data["lstmcell_output"])
            self.lstmcell_weight_hh = Tensor(data["lstmcell_weight_hh"])
            self.lstmcell_weight_ih = Tensor(data["lstmcell_weight_ih"])

    @weight(6.25)
    def test_linear_init(self):
        myLinearLayer = mp6.MyLinear(self.linear_input_size, self.linear_output_size)
        self.assertTrue(myLinearLayer.weight.shape == (self.linear_output_size, self.linear_input_size),
            'incorrect dimensions of weights of linear layer')
        self.assertTrue(myLinearLayer.bias.shape == (self.linear_output_size,),
            'incorrect dimensions of weights of linear layer')

    @weight(6.25)
    def test_linear_forward(self):
        myLinearLayer = mp6.MyLinear(self.linear_input_size, self.linear_output_size)
        myLinearLayer.weight = Parameter(Tensor(self.linear_weight))
        myLinearLayer.bias = Parameter(Tensor(self.linear_bias))
        myLinearOutput = myLinearLayer(self.linear_input)
        e=torchsum(torchabs(myLinearOutput-self.linear_output))/torchsum(torchabs(self.linear_output))
        self.assertTrue(e < 0.04, 'MyLinear\'s forward method wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_lstmcell_init(self):
        myLSTMCellLayer = mp6.MyLSTMCell(self.lstmcell_input_size, self.lstmcell_hidden_size)
        self.assertTrue(myLSTMCellLayer.weight_ih.shape == (4*self.lstmcell_hidden_size, self.lstmcell_input_size),
            'incorrect dimensions of weights of LSTM cell layer')
        self.assertTrue(myLSTMCellLayer.bias_ih.shape == (4*self.lstmcell_hidden_size,),
            'incorrect dimensions of biases of LSTM cell layer')
        self.assertTrue(myLSTMCellLayer.weight_hh.shape == (4*self.lstmcell_hidden_size, self.lstmcell_hidden_size),
            'incorrect dimensions of weights of LSTM cell layer')
        self.assertTrue(myLSTMCellLayer.bias_hh.shape == (4*self.lstmcell_hidden_size,),
            'incorrect dimensions of biases of LSTM cell layer')

    @weight(6.25)
    def test_lstmcell_forward(self):
        myLSTMCellLayer = mp6.MyLSTMCell(self.lstmcell_input_size, self.lstmcell_hidden_size)
        myLSTMCellLayer.weight_ih = Parameter(Tensor(self.lstmcell_weight_ih))
        myLSTMCellLayer.bias_ih = Parameter(Tensor(self.lstmcell_bias_ih))
        myLSTMCellLayer.weight_hh = Parameter(Tensor(self.lstmcell_weight_hh))
        myLSTMCellLayer.bias_hh = Parameter(Tensor(self.lstmcell_bias_hh))
        h_init = Tensor(self.lstmcell_h_init)
        c_init = Tensor(self.lstmcell_c_init)
        myLSTMCellOutput, myLSTMCellLayerCOut = myLSTMCellLayer(self.lstmcell_input, (h_init, c_init))
        e=torchsum(torchabs(myLSTMCellOutput-self.lstmcell_output))/torchsum(torchabs(self.lstmcell_output))
        self.assertTrue(e < 0.04, 'MyLSTMCell\'s forward method wrong by more than 4% (visible case)')
        e=torchsum(torchabs(myLSTMCellLayerCOut-self.lstmcell_cout))/torchsum(torchabs(self.lstmcell_cout))
        self.assertTrue(e < 0.04, 'MyLSTMCell\'s forward method wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_lstm_init(self):
        myLSTMLayer = mp6.LSTM(self.lstmcell_input_size, self.lstmcell_hidden_size)
        self.assertTrue(myLSTMLayer.hidden_size == self.lstmcell_hidden_size,
            'incorrect hidden size in LSTM layer')
        self.assertTrue(type(myLSTMLayer.bn) == BatchNorm1d,
            'bn member of LSTM class is not a BatchNorm1d')
        self.assertTrue(myLSTMLayer.bn.num_features == self.lstmcell_input_size,
            'incorrect feature size of bn member of LSTM class')
        self.assertTrue(myLSTMLayer.bn.affine == False,
            'bn member of LSTM class has learnable affine parameters')
        self.assertTrue(type(myLSTMLayer.linear) == mp6.MyLinear,
            'linear member of LSTM class is not a MyLinear')
        self.assertTrue(myLSTMLayer.linear.weight.shape == (self.lstmcell_input_size, self.lstmcell_hidden_size),
            'incorrect dimensions of weights of linear layer')
        self.assertTrue(myLSTMLayer.linear.bias.shape == (self.lstmcell_input_size,),
            'incorrect dimensions of weights of linear layer')
        self.assertTrue(type(myLSTMLayer.lstm) == mp6.MyLSTMCell,
            'lstm member of LSTM class is not a MyLSTM')
        self.assertTrue(myLSTMLayer.lstm.weight_ih.shape == (4*self.lstmcell_hidden_size, self.lstmcell_input_size),
            'incorrect dimensions of weights of lstm layer')
        self.assertTrue(myLSTMLayer.lstm.bias_ih.shape == (4*self.lstmcell_hidden_size,),
            'incorrect dimensions of biases of lstm layer')
        self.assertTrue(myLSTMLayer.lstm.weight_hh.shape == (4*self.lstmcell_hidden_size, self.lstmcell_hidden_size),
            'incorrect dimensions of weights of lstm layer')
        self.assertTrue(myLSTMLayer.lstm.bias_hh.shape == (4*self.lstmcell_hidden_size,),
            'incorrect dimensions of biases of lstm layer')

    @weight(6.25)
    def test_lstm_weightinit(self):
        myLSTMLayer = mp6.LSTM(self.lstmcell_input_size, self.lstmcell_hidden_size)
        myLSTMLayer.initWeight(self.lstm_forget_bias)
        for bias, bias_name in zip([myLSTMLayer.lstm.bias_ih, myLSTMLayer.lstm.bias_hh],['bias_ih', 'bias_hh']):
            biases_ih_i = bias[0:self.lstmcell_hidden_size]
            biases_ih_f = bias[self.lstmcell_hidden_size:2*self.lstmcell_hidden_size]
            biases_ih_c = bias[2*self.lstmcell_hidden_size:3*self.lstmcell_hidden_size]
            biases_ih_o = bias[3*self.lstmcell_hidden_size:4*self.lstmcell_hidden_size]
            self.assertTrue((biases_ih_i == 0).all(),
                bias_name+' for input gate not zeroed out')
            self.assertTrue((biases_ih_f == self.lstm_forget_bias).all(),
                bias_name+' for forget gate not set to provided bias')
            self.assertTrue((biases_ih_c == 0).all(),
                bias_name+' for cell gate not zeroed out')
            self.assertTrue((biases_ih_o == 0).all(),
                bias_name+' for output gate not zeroed out')
        self.assertTrue((myLSTMLayer.linear.bias == 0).all(),
            bias_name+' for linear layer not set to provided bias')

    @weight(6.25)
    def test_lstm_hiddeninit(self):
        myLSTMLayer = mp6.LSTM(self.lstmcell_input_size, self.lstmcell_hidden_size)
        myLSTMLayer.initHidden(self.lstm_batch_size)
        self.assertTrue(count_nonzero(myLSTMLayer.h) == 0,
            'Nonzero values in initialized hidden state')
        self.assertTrue(count_nonzero(myLSTMLayer.c) == 0,
            'Nonzero values in initialized memory cell')

    @weight(6.25)
    def test_lstm_forward(self):
        myLSTMLayer = mp6.LSTM(self.lstmcell_input_size, self.lstmcell_hidden_size)
        myLSTMLayer.lstm.weight_ih = Parameter(Tensor(self.lstmcell_weight_ih))
        myLSTMLayer.lstm.bias_ih = Parameter(Tensor(self.lstmcell_bias_ih))
        myLSTMLayer.lstm.weight_hh = Parameter(Tensor(self.lstmcell_weight_hh))
        myLSTMLayer.lstm.bias_hh = Parameter(Tensor(self.lstmcell_bias_hh))
        myLSTMLayer.linear.weight = Parameter(Tensor(self.lstm_linear_weight))
        myLSTMLayer.linear.bias = Parameter(Tensor(self.lstm_linear_bias))
        myLSTMLayer.h = Variable(self.lstm_h)
        myLSTMLayer.c = Variable(self.lstm_c)
        myLSTMOutput = myLSTMLayer(self.lstm_inputs, self.lstm_n_frames)
        e=torchsum(torchabs(myLSTMOutput-self.lstm_output))/torchsum(torchabs(self.lstm_output))
        self.assertTrue(e < 0.04, 'MyLSTM\'s forward method wrong by more than 4% (visible case)')

