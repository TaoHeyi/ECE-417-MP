import unittest, h5py, mp4
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

class TestStep(unittest.TestCase):
    def setUp(self):
        self.h5 = h5py.File('solutions.hdf5','r')
        self.Lambda = mp4.HiddenMarkovModel(self.h5['A'],self.h5['mu'],self.h5['var'])
        self.transcript=[]
        with open('data/LDC93S1.phn') as f:
            for line in f:
                fields = line.strip().split()
                if len(fields)>1:
                    self.transcript.append(fields[2])
        self.phn2idx = {}
        for p in self.transcript:
            if p not in self.phn2idx:
                self.phn2idx[p]=len(self.phn2idx)

    @weight(6.25)
    def test_Quniform(self):
        Quniform = mp4.todo_Quniform(self.transcript, self.phn2idx, self.h5['X'].shape[1])
        e=np.sum(np.abs(Quniform-self.h5['Quniform']))/np.sum(np.abs(self.h5['Quniform']))
        self.assertTrue(e < 0.04, 'todo_Quniform wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_Lambda(self):
        Lambda = mp4.todo_Lambda(self.h5['Quniform'][:], self.h5['X'])
        e=np.sum(np.abs(Lambda.A-self.h5['A']))/np.sum(np.abs(self.h5['A']))
        self.assertTrue(e < 0.04, 'todo_Lambda.A wrong by more than 4% (visible case)')
        e=np.sum(np.abs(Lambda.mu-self.h5['mu']))/np.sum(np.abs(self.h5['mu']))
        self.assertTrue(e < 0.04, 'todo_Lambda.mu wrong by more than 4% (visible case)')
        e=np.sum(np.abs(Lambda.var-self.h5['var']))/np.sum(np.abs(self.h5['var']))
        self.assertTrue(e < 0.04, 'todo_Lambda.var wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_Bscaled(self):
        logB, Bscaled = mp4.todo_Bscaled(self.h5['X'], self.Lambda)
        e=np.sum(np.abs(Bscaled-self.h5['Bscaled']))/np.sum(np.abs(self.h5['Bscaled']))
        self.assertTrue(e < 0.04, 'todo_Bscaled wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_Qstar(self):
        Qstar = mp4.todo_Qstar(self.h5['psi'])
        e=np.sum(np.abs(Qstar-self.h5['Qstar']))/np.sum(np.abs(self.h5['Qstar']))
        self.assertTrue(e < 0.04, 'todo_Qstar wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_alphahat(self):
        alphahat, G = mp4.todo_alphahat(self.h5['Bscaled'], self.Lambda)
        e=np.sum(np.abs(alphahat-self.h5['alphahat']))/np.sum(np.abs(self.h5['alphahat']))
        self.assertTrue(e < 0.04, 'todo_alphahat wrong by more than 4% (visible case)')
        e=np.sum(np.abs(G-self.h5['G']))/np.sum(np.abs(self.h5['G']))
        self.assertTrue(e < 0.04, 'G wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_betahat(self):
        betahat = mp4.todo_betahat(self.h5['Bscaled'], self.Lambda)
        e=np.sum(np.abs(betahat-self.h5['betahat']))/np.sum(np.abs(self.h5['betahat']))
        self.assertTrue(e < 0.04, 'todo_betahat wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_xi(self):
        alphahat = self.h5['alphahat'][:]
        betahat = self.h5['betahat'][:]
        Bscaled = self.h5['Bscaled'][:]
        xi = mp4.todo_xi(alphahat,betahat,Bscaled,self.Lambda)
        e=np.sum(np.abs(xi-self.h5['xi']))/np.sum(np.abs(self.h5['xi']))
        self.assertTrue(e < 0.04, 'todo_xi wrong by more than 4% (visible case)')
        
    @weight(6.25)
    def test_Lambdaprime(self):
        Lambdaprime = mp4.todo_Lambdaprime(self.h5['xi'], self.h5['X'])
        e=np.sum(np.abs(Lambdaprime.A-self.h5['Aprime']))/np.sum(np.abs(self.h5['Aprime']))
        self.assertTrue(e < 0.04, 'todo_Lambdaprime.A wrong by more than 4% (visible case)')
        e=np.sum(np.abs(Lambdaprime.mu-self.h5['muprime']))/np.sum(np.abs(self.h5['muprime']))
        self.assertTrue(e < 0.04, 'todo_Lambdaprime.mu wrong by more than 4% (visible case)')
        e=np.sum(np.abs(Lambdaprime.var-self.h5['varprime']))/np.sum(np.abs(self.h5['varprime']))
        self.assertTrue(e < 0.04, 'todo_Lambdaprime.var wrong by more than 4% (visible case)')

