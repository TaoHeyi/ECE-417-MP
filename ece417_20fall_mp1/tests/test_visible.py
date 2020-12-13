import unittest, h5py, mp1
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    def setUp(self):
        self.h5 = h5py.File('solutions.hdf5','r')

    @weight(6.25)
    def test_dataset_mean(self):
        train = mp1.matrix_and_vector_to_dataset(self.h5['trainmatrix'], self.h5['trainlabelvec'])
        mu = mp1.todo_dataset_mean(train)
        e = np.sum(np.abs(mu-self.h5['mu']))/np.sum(np.abs(self.h5['mu']))
        self.assertTrue(e < 0.04, 'todo_dataset_mean wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_center_datasets(self):
        train = mp1.matrix_and_vector_to_dataset(self.h5['trainmatrix'], self.h5['trainlabelvec'])
        dev = mp1.matrix_and_vector_to_dataset(self.h5['devmatrix'], self.h5['devlabelvec'])
        test = mp1.matrix_and_vector_to_dataset(self.h5['testmatrix'], self.h5['testlabelvec'])
        mu = self.h5['mu']
        ctrain, cdev, ctest = mp1.todo_center_datasets(train, dev, test, mu)
        ctrain_ref = self.h5['ctrainmatrix']
        etrain = np.sum(np.abs(ctrain_ref-mp1.dataset_to_matrix(ctrain)))/np.sum(np.abs(ctrain_ref))
        self.assertTrue(etrain < 0.04, 'todo_center_datasets ctrain wrong by more than 4% (visible case)')
        cdev_ref = self.h5['cdevmatrix']
        edev = np.sum(np.abs(cdev_ref-mp1.dataset_to_matrix(cdev)))/np.sum(np.abs(cdev_ref))
        self.assertTrue(edev < 0.04, 'todo_center_datasets cdev wrong by more than 4% (visible case)')
        ctest_ref = self.h5['ctestmatrix']
        etest = np.sum(np.abs(ctest_ref-mp1.dataset_to_matrix(ctest)))/np.sum(np.abs(ctest_ref))
        self.assertTrue(etest < 0.04, 'todo_center_datasets ctest wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_find_transform(self):
        ctrain = mp1.matrix_and_vector_to_dataset(self.h5['ctrainmatrix'], self.h5['ctrainlabelvec'])
        transform, variances = mp1.todo_find_transform(ctrain)
#        etransform = np.sum(np.abs(transform-self.h5['transform']))/np.sum(np.abs(self.h5['transform']))
#        self.assertTrue(etransform < 0.04, 'todo_find_transform transform wrong by %g which is more than 4% (visible case)'%(etransform))
        evariances = np.sum(np.abs(variances-self.h5['variances']))/np.sum(np.abs(self.h5['variances']))
        self.assertTrue(evariances < 0.04, 'todo_find_transform variances wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_transform_datasets(self):
        ctrain = mp1.matrix_and_vector_to_dataset(self.h5['ctrainmatrix'], self.h5['ctrainlabelvec'])
        cdev = mp1.matrix_and_vector_to_dataset(self.h5['cdevmatrix'], self.h5['cdevlabelvec'])
        ctest = mp1.matrix_and_vector_to_dataset(self.h5['ctestmatrix'], self.h5['ctestlabelvec'])
        transform = self.h5['transform']
        ttrain, tdev, ttest = mp1.todo_transform_datasets(ctrain, cdev, ctest, transform)
        ttrain_ref = self.h5['ttrainmatrix']
        etrain = np.sum(np.abs(ttrain_ref-mp1.dataset_to_matrix(ttrain)))/np.sum(np.abs(ttrain_ref))
        self.assertTrue(etrain < 0.04, 'todo_transform_datasets ttrain wrong by more than 4% (visible case)')
        tdev_ref = self.h5['tdevmatrix']
        edev = np.sum(np.abs(tdev_ref-mp1.dataset_to_matrix(tdev)))/np.sum(np.abs(tdev_ref))
        self.assertTrue(edev < 0.04, 'todo_transform_datasets tdev wrong by more than 4% (visible case)')
        ttest_ref = self.h5['ttestmatrix']
        etest = np.sum(np.abs(ttest_ref-mp1.dataset_to_matrix(ttest)))/np.sum(np.abs(ttest_ref))
        self.assertTrue(etest < 0.04, 'todo_transform_datasets ttest wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_distances(self):
        ttrain = mp1.matrix_and_vector_to_dataset(self.h5['ttrainmatrix'], self.h5['ttrainlabelvec'])
        tdev = mp1.matrix_and_vector_to_dataset(self.h5['tdevmatrix'], self.h5['tdevlabelvec'])
        fullsize = len(ttrain)
        Dfull = mp1.todo_distances(ttrain, tdev, fullsize)
        e = np.sum(np.abs(Dfull-self.h5['Dfull']))/np.sum(np.abs(self.h5['Dfull']))
        self.assertTrue(e < 0.04, 'todo_test_distances dev D, fullsize, wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_nearest_neighbor(self):
        ttrain = mp1.matrix_and_vector_to_dataset(self.h5['ttrainmatrix'], self.h5['ttrainlabelvec'])
        Dfull = self.h5['Dfull']
        hypsfull = mp1.todo_nearest_neighbor(ttrain, Dfull)
        e = np.sum(np.abs(hypsfull-self.h5['hypsfull']))/np.sum(np.abs(self.h5['hypsfull']))
        self.assertTrue(e < 0.04, 'todo_nearest_neighbor dev hyps, fullsize, wrong by more than 4% (visible case) ')

    @weight(6.25)
    def test_compute_accuracy(self):
        tdev = mp1.matrix_and_vector_to_dataset(self.h5['tdevmatrix'], self.h5['tdevlabelvec'])
        refs = [ datum.y  for datum in tdev ]
        hypsfull = self.h5['hypsfull']
        accuracyfull, confusionfull = mp1.todo_compute_accuracy(tdev, hypsfull)
        e = np.sum(np.abs(confusionfull-self.h5['confusionfull']))/np.sum(np.abs(self.h5['confusionfull']))
        self.assertTrue(e<0.04, 'todo_compute_accuracy dev confusion, fullsize, wrong by more than 4% (visible case)')

    @weight(6.25)
    def test_find_bestsize(self):
        ttrain = mp1.matrix_and_vector_to_dataset(self.h5['ttrainmatrix'], self.h5['ttrainlabelvec'])
        tdev = mp1.matrix_and_vector_to_dataset(self.h5['tdevmatrix'], self.h5['tdevlabelvec'])
        variances = self.h5['variances']
        bestsize, accuracies = mp1.todo_find_bestsize(ttrain, tdev, variances)
        e = np.sum(np.abs(accuracies - self.h5['accuracies']))/np.sum(np.abs(self.h5['accuracies']))
        self.assertTrue(e<0.04, 'todo_find_bestsize accuracies wrong by more than 4% (visible case)')
