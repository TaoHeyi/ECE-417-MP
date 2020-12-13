import unittest, h5py, mp2
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    def setUp(self):
        self.datadir='data'
        self.h5 = h5py.File('solutions.hdf5','r')
        
    @weight(6.25)
    def test_spectrograms(self):
        nwtrain = self.h5['nwtrain']
        nwdev = self.h5['nwdev']
        nwtest = self.h5['nwtest']
        sgtrain,sgdev,sgtest=mp2.todo_spectrograms(nwtrain,nwdev,nwtest,16000,0.025,0.01,1024)
        e = np.sum(np.abs(sgtrain-self.h5['sgtrain']))/np.sum(np.abs(self.h5['sgtrain']))
        self.assertTrue(e < 0.04, 'todo_spectrograms sgtrain wrong by more than 4% (visible case)')
        e = np.sum(np.abs(sgdev-self.h5['sgdev']))/np.sum(np.abs(self.h5['sgdev']))
        self.assertTrue(e < 0.04, 'todo_spectrograms sgdev wrong by more than 4% (visible case)')
        e = np.sum(np.abs(sgtest-self.h5['sgtest']))/np.sum(np.abs(self.h5['sgtest']))
        self.assertTrue(e < 0.04, 'todo_spectrograms sgtest wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_melfilters(self):
        sgtrain = self.h5['sgtrain']
        melfilters, hertzmelcenters = mp2.todo_melfilters(sgtrain.shape[1],40,16000)
        e = np.sum(np.abs(melfilters-self.h5['melfilters']))/np.sum(np.abs(self.h5['melfilters']))
        self.assertTrue(e < 0.04, 'todo_melfilters melfilters wrong by more than 4% (visible case)')
        e = np.sum(np.abs(hertzmelcenters-self.h5['hertzmelcenters']))/np.sum(np.abs(self.h5['hertzmelcenters']))
        self.assertTrue(e < 0.04, 'todo_melfilters hertzmelcenters wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_filterbank(self):
        sgtrain = self.h5['sgtrain']
        sgdev = self.h5['sgdev']
        sgtest = self.h5['sgtest']
        melfilters = self.h5['melfilters']
        fbtrain,fbdev,fbtest=mp2.todo_filterbank(sgtrain,sgdev,sgtest,melfilters)
        e = np.sum(np.abs(fbtrain-self.h5['fbtrain']))/np.sum(np.abs(self.h5['fbtrain']))
        self.assertTrue(e < 0.04, 'todo_filterbank fbtrain wrong by more than 4% (visible case)')
        e = np.sum(np.abs(fbdev-self.h5['fbdev']))/np.sum(np.abs(self.h5['fbdev']))
        self.assertTrue(e < 0.04, 'todo_filterbank fbdev wrong by more than 4% (visible case)')
        e = np.sum(np.abs(fbtest-self.h5['fbtest']))/np.sum(np.abs(self.h5['fbtest']))
        self.assertTrue(e < 0.04, 'todo_filterbank fbtest wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_gtfilters(self):
        gtfilters, hertzerbcenters = mp2.todo_gtfilters(40, 16000, 0.025)
        e = np.sum(np.abs(gtfilters-self.h5['gtfilters']))/np.sum(np.abs(self.h5['gtfilters']))
        self.assertTrue(e < 0.04, 'todo_melfilters melfilters wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_gammatone(self):
        nwtrain = self.h5['nwtrain']
        nwdev = self.h5['nwdev']
        nwtest = self.h5['nwtest']
        gtfilters = self.h5['gtfilters']
        gttrain,gtdev,gttest=mp2.todo_gammatone(nwtrain,nwdev,nwtest,gtfilters,16000,0.01)
        e = np.sum(np.abs(gttrain-self.h5['gttrain']))/np.sum(np.abs(self.h5['gttrain']))
        self.assertTrue(e < 0.04, 'todo_gammatone gttrain wrong by more than 4% (visible case)')
        e = np.sum(np.abs(gtdev-self.h5['gtdev']))/np.sum(np.abs(self.h5['gtdev']))
        self.assertTrue(e < 0.04, 'todo_gammatone gtdev wrong by more than 4% (visible case)')
        e = np.sum(np.abs(gttest-self.h5['gttest']))/np.sum(np.abs(self.h5['gttest']))
        self.assertTrue(e < 0.04, 'todo_gammatone gttest wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_distances(self):
        fbtrain = self.h5['fbtrain']
        fbdev = self.h5['fbdev']
        gttrain = self.h5['gttrain']
        gtdev = self.h5['gtdev']
        fbdist = mp2.todo_distances(fbtrain, fbdev)
        gtdist = mp2.todo_distances(fbtrain, fbdev)
        e = np.sum(np.abs(fbdist-self.h5['fbdist']))/np.sum(np.abs(self.h5['fbdist']))
        self.assertTrue(e < 0.04, 'todo_distances filterbank wrong by more than 4% (visible case)')
        e = np.sum(np.abs(gtdist-self.h5['gtdist']))/np.sum(np.abs(self.h5['gtdist']))
        self.assertTrue(e < 0.04, 'todo_distances gammatone wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_hyps(self):
        train, dev, test, labels = mp2.load_datasets(self.datadir)        
        fbdist = self.h5['fbdist']
        gtdist = self.h5['gtdist']
        fbhyps = mp2.todo_nearest_neighbor(train, fbdist)
        gthyps = mp2.todo_nearest_neighbor(train, gtdist)
        e = np.sum(np.abs(fbhyps-self.h5['fbhyps']))/np.sum(np.abs(self.h5['fbhyps']))
        self.assertTrue(e < 0.04, 'todo_nearest_neighbor filterbank wrong by more than 4% (visible case)')
        e = np.sum(np.abs(gthyps-self.h5['gthyps']))/np.sum(np.abs(self.h5['gthyps']))
        self.assertTrue(e < 0.04, 'todo_nearest_neighbor gammatone wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_confusion(self):
        train, dev, test, labels = mp2.load_datasets(self.datadir)        
        fbhyps = self.h5['fbhyps']
        fbacc, fbconfusion = mp2.todo_compute_accuracy(dev, fbhyps)        
        e = np.sum(np.abs(fbconfusion-self.h5['fbconfusion']))/np.sum(np.abs(self.h5['fbconfusion']))
        self.assertTrue(e < 0.04, 'todo_compute_accuracy fb confusions wrong by more than 4% (visible)')
        gthyps = self.h5['gthyps']
        gtacc, gtconfusion = mp2.todo_compute_accuracy(dev, gthyps)        
        e = np.sum(np.abs(gtconfusion-self.h5['gtconfusion']))/np.sum(np.abs(self.h5['gtconfusion']))
        self.assertTrue(e < 0.04, 'todo_compute_accuracy gt confusions wrong by more than 4% (visible)')
