import unittest, h5py, mp3
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

class TestStep(unittest.TestCase):
    def setUp(self):
        self.datadir='data'
        self.h5 = h5py.File('solutions.hdf5','r')
        self.w = h5py.File('weights_trained.hdf5','r')
        
    @weight(7.14)
    def test_concatenation(self):
        concatenation = mp3.todo_concatenate(self.h5['features'])
        e=np.sum(np.abs(concatenation-self.h5['concatenation']))/np.sum(np.abs(self.h5['concatenation']))
        self.assertTrue(e < 0.04, 'todo_concatenation wrong by more than 4% (visible case)')

    @weight(7.15)
    def test_forward(self):
        hypothesis,hidden = mp3.todo_forward(self.h5['concatenation'],self.w['W1'],self.w['W2'])
        e=np.sum(np.abs(hypothesis-self.h5['hypothesis']))/np.sum(np.abs(self.h5['hypothesis']))
        self.assertTrue(e < 0.04, 'todo_forward hypothesis wrong by more than 4% (visible case)')
        e=np.sum(np.abs(hidden-self.h5['hidden']))/np.sum(np.abs(self.h5['hidden']))
        self.assertTrue(e < 0.04, 'todo_forward hidden wrong by more than 4% (visible case)')

    @weight(7.14)
    def test_detect_rectangles(self):
        dataset = mp3.MP3_Dataset(self.datadir)        
        best_rects = mp3.todo_detect_rectangles(self.h5['hypothesis'], 10, dataset)
        e=np.sum(np.abs(best_rects-self.h5['best_rects']))/np.sum(np.abs(self.h5['best_rects']))
        self.assertTrue(e < 0.04, 'todo_detect_rectangles wrong by more than 4% (visible case)')

    @weight(7.14)
    def test_outputgrad(self):
        outputgrad = mp3.todo_outputgrad(self.h5['hypothesis'], self.h5['target'])
        e=np.sum(np.abs(outputgrad-self.h5['outputgrad']))/np.sum(np.abs(self.h5['outputgrad']))
        self.assertTrue(e < 0.04, 'todo_outputgrad wrong by more than 4% (visible case)')

    @weight(7.14)
    def test_backprop(self):
        backprop = mp3.todo_backprop(self.h5['outputgrad'][:],self.h5['hidden'][:],self.w['W2'][:])
        e=np.sum(np.abs(backprop-self.h5['backprop']))/np.sum(np.abs(self.h5['backprop']))
        self.assertTrue(e < 0.04, 'todo_backprop wrong by more than 4% (visible case)')

    @weight(7.15)
    def test_weightgrad(self):
        dW1,dW2 = mp3.todo_weightgrad(self.h5['outputgrad'][:], self.h5['backprop'][:], self.h5['hidden'][:], self.h5['concatenation'][:])
        e=np.sum(np.abs(dW1-self.h5['dW1']))/np.sum(np.abs(self.h5['dW1']))
        self.assertTrue(e < 0.04, 'todo_weightgrad dW1 wrong by more than 4% (visible case)')
        e=np.sum(np.abs(dW2-self.h5['dW2']))/np.sum(np.abs(self.h5['dW2']))
        self.assertTrue(e < 0.04, 'todo_weightgrad dW2 wrong by more than 4% (visible case)')

    @weight(7.14)
    def test_update_weights(self):
        W1,W2 = mp3.todo_update_weights(self.w['W1'][:],self.w['W2'][:],self.h5['dW1'][:],self.h5['dW2'][:],0.01)
        e=np.sum(np.abs(W1-self.h5['W1']))/np.sum(np.abs(self.h5['W1']))
        self.assertTrue(e < 0.04, 'todo_weight_update new_W1 wrong by more than 4% (visible case)')
        e=np.sum(np.abs(W2-self.h5['W2']))/np.sum(np.abs(self.h5['W2']))
        self.assertTrue(e < 0.04, 'todo_weight_update new_W2 wrong by more than 4% (visible case)')

