import os, h5py
from PIL import Image
import numpy as  np

###############################################################################
# First, some utility functions to help you load and convert image data
class Datum(object):
    def __init__(self, imagepath=None, personname=None, personnames=None, x=None, y=None):
        '''Load self.x=vectorized image, self.y=integer index of personname'''
        if imagepath is not None:
            image = np.asarray(Image.open(imagepath)).astype('float64')     #store the image from the given datapath as a float
            self.x = image.reshape(np.prod(image.shape))                    #
        if personname is not None and personnames is not None:
            self.y = personnames.index(personname)
        if x is not None:
            self.x = x.copy()
        if y is not None:
            self.y = y

def load_datasets(datapath):
    '''Load image datasets; divide into train, dev, and test subsets'''
    train, dev, test = [], [], []
    personnames = os.listdir(datapath)                                      #list of persionnames in a tuple
    n = 0
    for personname in personnames:                                          #use to list all the folders and files
        for filename in os.listdir(os.path.join(datapath,personname)):
            imagepath = os.path.join(datapath,personname,filename)
            if n % 5 == 4:
                test.append(Datum(imagepath, personname, personnames))
            elif n % 5 == 3:
                dev.append(Datum(imagepath, personname, personnames))
            else:
                train.append(Datum(imagepath, personname, personnames))
            n += 1
    return(train,dev,test,personnames)

def copy_datasets(train, dev, test):
    '''Make deep copies of the three datasets'''
    new_train = [ Datum(x=datum.x,y=datum.y) for datum in train ]
    new_dev = [ Datum(x=datum.x,y=datum.y) for datum in dev ]
    new_test = [ Datum(x=datum.x,y=datum.y) for datum in test ]
    return(new_train, new_dev, new_test)

def vector_to_image(vector, shape=(250,250,3)):
    '''Convert a vector to an image (with shape (nrows, ncols, ncolors)'''
    return(vector.reshape(shape))

def dataset_to_matrix(dataset):
    '''Create a matrix in which each row is a vectorized observation from the dataset'''
    return(np.array([datum.x for datum in dataset]))

def dataset_to_labelvec(dataset):
    '''Create a vector in which each element is a label from the dataset'''
    return(np.array([datum.y for datum in dataset], dtype='int'))

def matrix_and_vector_to_dataset(matrix, labelvec):
    '''Create a dataset from a data matrix and label vector'''
    dataset = [ Datum(x=matrix[i,:],y=labelvec[i]) for i in range(len(labelvec)) ]
    return(dataset)

###############################################################################
# TODO: here are the functions that you need to write
def todo_dataset_mean(dataset):
    '''Compute mu = the average x from the provided dataset'''
    matrix_ds = dataset_to_matrix(dataset)
    a = np.sum(matrix_ds, axis=0)
    shape_a = matrix_ds.shape
    rows_a = shape_a[0]
    mu = a/rows_a
    return(mu)

def todo_center_datasets(train, dev, test, mu):
    '''
    Deep-copy train, dev, test (deep copy helps debugging;
    it's not good memory-management practice in general).
    Subtract mu from each x, return the resulting three datasets.
    '''
    ctrain, cdev, ctest = copy_datasets(train, dev, test)
    for datum in ctrain:
        datum.x = datum.x - mu
    for datum in cdev:
        datum.x = datum.x - mu
    for datum in ctest:
        datum.x = datum.x - mu
    return(ctrain,cdev,ctest)

def todo_find_transform(dataset):
    '''
    Find and return the PCA transform for the given dataset:
    a matrix in which each column is a principal component direction.
    You can assume that the # data is less than the # dimensions per vector,
    so you should probably use the gram-matrix method, not the covariance
    method.
    Standardization: Make sure that each of your returned vectors has unit norm,
    and that its first element is non-negative.
    Return: (transform, variances)
      transform[:,i] = the i'th principal component direction
      variances[i] = the variance explained by the i'th principal component
    '''
    X = dataset_to_matrix(dataset)
    XXT = np.dot(X,X.T)
    E,U = np.linalg.eig(XXT)
    V_u = np.dot(X.T, U)
    V_normed = V_u / V_u.max(axis=0)
    return(V_normed, E)

def todo_transform_datasets(ctrain, cdev, ctest, transform):
    '''
    Deep-copy train, dev, test (deep copy helps debugging;
    it's not good memory-management practice in general).
    Transform each x using transform, return the resulting three datasets.
    '''
    ttrain, tdev, ttest = copy_datasets(ctrain, cdev, ctest)
    ttrain1 = dataset_to_matrix(ttrain)
    tdev1 = dataset_to_matrix(tdev)
    ttest1 = dataset_to_matrix(ttest)
    new_ttrain = np.dot(ttrain1, transform)
    new_tdev = np.dot(tdev1, transform)
    new_ttest = np.dot(ttest1, transform)
    for i in range(len(ttrain)):
        ttrain[i].x = new_ttrain[i]
    for i in range(len(tdev)):
        tdev[i].x = new_tdev[i]
    for i in range(len(ttest)):
        ttest[i].x = new_ttest[i]
    return(ttrain, tdev, ttest)

def todo_distances(train,test,size):
    '''
    Return a matrix D such that D[i,j]=distance(train[i].x[:size],test[j].x[:size])
    '''
    D = np.zeros(shape=(len(train), len(test)))
    for i in range(len(train)):
        for j in range(len(test)):
            D[i,j] = np.linalg.norm(train[i].x[:size] - test[j].x[:size])
    return(D)

def todo_nearest_neighbor(train, D):
    '''
    Given the dataset train, and the (NTRAINxNTEST) matrix D, returns
    an int numpy array of length NTEST, specifying the person number (y) of the training token
    that is closest to each of the NTEST test tokens.
    '''
    #(ntrain, ndev) = D.shape
    #hyps = np.zeros(ndev, dtype='int')
    closest = np.argmin(D, axis=0)
    hyps = np.array([train[index].y for index in closest])
    #print(hyps)
    return(hyps)

def todo_compute_accuracy(test, hyps):
    '''
    Compute accuracy, confusion.
    confusion is a (4x4) matrix.
    confusion[ref,hyp] is the number of test tokens for which test[i].y==ref
    but hyps[i]==hyp.
    accuracy is a scalar between 0 and 1, equal to the total fraction of
    hyps that are correct.
    '''
    confusion = np.zeros((4,4))
    total = len(hyps)
    for i in range(len(hyps)):
        ref = test[i].y
        hyp = hyps[i]
        confusion[ref, hyp] += 1
    num = 0
    for i in range(4):
        num += confusion[i,i]
    accuracy = num / total
    return(accuracy, confusion)

def todo_find_bestsize(train, dev, variances):
    '''
    Find and return (bestsize, accuracies):
    accuracies = accuracy of dev classification, as function of
    PCA feature vector dimension.
    The only dimensions you need to test (the only nonzero entries in this
    vector) are the ones where the PCA features explain between 92.5% and
    97.5% of the variance of the training set, as specified by the provided
    per-feature variances.  All others should be zero.
    bestsize = argmax(accuracies)
    '''
    accuracies = np.zeros(len(train))
    for i in range(len(variances)):
        energy_per = np.sum(variances[0:i + 1]) / np.sum(variances)
        if(energy_per>0.925 and energy_per<0.975):
            size = i + 1
            distance = todo_distances(train, dev, size)
            hyps = todo_nearest_neighbor(train, distance)
            accuracy, confusion = todo_compute_accuracy(dev, hyps)
            accuracies[i] = accuracy
    bestsize = np.argmax(accuracies)
    return(bestsize, accuracies)

###############################################################################
# Here is the code that gets called if you type python mp1.py on command line.
# For each step of processing:
#  1. do the step
#  2. pop up a window to show the result  in a figure.
#  3. wait for you to close the window, then continue.
# Almost all of the code below is just creating the figures to show you.
# None of the code below will get run by the autograder;
# if you want to see what the autograder does, type python run_tests.py
# on the command line.
if __name__=="__main__":
    import tkinter as tk
    from matplotlib.backend_bases import key_press_handler
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.figure, argparse
    parser = argparse.ArgumentParser(
        description='''Run MP1 using provided data directory, and show popup windows of results.''')
    parser.add_argument('-s','--steps',nargs='*',
                        help='''Perform only the specified steps (should be a space-separated list).
                        Results of any preceding steps are read from solutions.hdf5.
                        This is intended so that you can focus debugging on the step that you specify.
                        Steps are: 0: load_datasets, 1: todo_dataset_mean
                        2: todo_center_datasets,
                        3: todo_find_transform,
                        4: todo_transform_datasets,
                        5: todo_distances,
                        6: todo_nearest_neighbor,
                        7: todo_compute_accuracy,
                        8: todo_find_bestsize.
                        Default: do all steps''')
    args = parser.parse_args()
    if args.steps is not None:
        solutions = h5py.File('solutions.hdf5','r')

    class PlotWindow(object):
        '''
        Pop up a window containing a matplotlib Figure.
        The NavigationToolbar2TK allows you to save the figure in a file.
        The key_press_handler  permits standard key events as described at
        https://matplotlib.org/3.3.0/users/interactive.html#key-event-handling
        '''
        def __init__(self, fig):
            import tkinter
            self.root = tkinter.Tk()
            self.canvas = FigureCanvasTkAgg(fig, master=self.root)
            toolbar = NavigationToolbar2Tk(self.canvas, self.root)
            self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
            self.canvas.mpl_connect("key_press_event", lambda e: key_press_handler(e,self.canvas,toolbar))
            button=tkinter.Button(master=self.root, text="Close and continue", command=self.quit)
            button.pack(side=tkinter.BOTTOM)
            self.root.mainloop()
        def quit(self):
            self.root.quit()     # stops mainloop
            self.root.destroy()  # this is necessary on Windows

    ##############################################################
    # Step 0a: load the data, and show a scatter plot of train set
    if args.steps is None or '0' in args.steps:
        train, dev, test, personnames = load_datasets('data')               #load all the files in the data
        mat = dataset_to_matrix(train)
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(2,2,sharex=True)
        axs[0,0].plot(mat[:,0],mat[:,1],'.')
        axs[0,0].set_title('Pixel 1 vs. Pixel 0')
        axs[0,1].plot(mat[:,0],mat[:,2],'.')
        axs[0,1].set_title('Pixel 2 vs. Pixel 0')
        axs[1,0].plot(mat[:,0],mat[:,3],'.')
        axs[1,0].set_title('Pixel 3 vs. Pixel 0')
        axs[1,1].plot(mat[:,0],mat[:,4],'.')
        axs[1,1].set_title('Pixel 4 vs. Pixel 0')
        PlotWindow(fig)

        # Step 0b: show histograms of train set
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(2,2,sharex=True)
        axs[0,0].hist(mat[:,0],bins=100)
        axs[0,0].set_title('Pixel 0 histogram')
        axs[0,0].set_ylabel('# images')
        axs[0,1].hist(mat[:,1],bins=100)
        axs[0,1].set_title('Pixel 1 histogram')
        axs[1,0].hist(mat[:,2],bins=100)
        axs[1,0].set_title('Pixel 2 histogram')
        axs[1,0].set_xlabel('Pixel value')
        axs[1,0].set_ylabel('# images')
        axs[1,1].hist(mat[:,3],bins=100)
        axs[1,1].set_title('Pixel 3 histogram')
        axs[1,1].set_xlabel('Pixel value')
        PlotWindow(fig)
    else:
        personnames = os.listdir('data')
        train = matrix_and_vector_to_dataset(solutions['trainmatrix'], solutions['trainlabelvec'])
        dev = matrix_and_vector_to_dataset(solutions['devmatrix'], solutions['devlabelvec'])
        test = matrix_and_vector_to_dataset(solutions['testmatrix'], solutions['testlabelvec'])

    ##############################################################
    # Step 1: Compute a mean vector
    if args.steps is None or '1' in args.steps:
        mu = todo_dataset_mean(train)
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        ax = fig.subplots(1,1)
        ax.clear()
        ax.imshow(vector_to_image(mu).astype(dtype='uint8'))
        ax.set_title('Average of all training images')
        #ax.imshow(vector_to_image(mu))
        PlotWindow(fig)
    else:
        mu = solutions['mu']

    ###############################################################
    # Step 2a: Center all three center_datasets
    if args.steps is None or '2' in args.steps:
        ctrain, cdev, ctest = todo_center_datasets(train, dev, test, mu)
        cmat = dataset_to_matrix(ctrain)
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(2,2,sharex=True)
        axs[0,0].clear()
        axs[0,0].plot(cmat[:,0],cmat[:,1],'.')
        axs[0,0].set_title('Centered Pixel 1 vs. Pixel 0')
        axs[0,1].clear()
        axs[0,1].plot(cmat[:,0],cmat[:,2],'.')
        axs[0,1].set_title('Centered Pixel 2 vs. Pixel 0')
        axs[1,0].clear()
        axs[1,0].plot(cmat[:,0],cmat[:,3],'.')
        axs[1,0].set_title('Centered Pixel 3 vs. Pixel 0')
        axs[1,1].clear()
        axs[1,1].plot(cmat[:,0],cmat[:,4],'.')
        axs[1,1].set_title('Centered Pixel 4 vs. Pixel 0')
        PlotWindow(fig)

        # Step 2b: show histograms of centered train set
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(2,2,sharex=True)
        axs[0,0].hist(cmat[:,0],bins=100)
        axs[0,0].set_title('Centered 0 histogram')
        axs[0,0].set_ylabel('# images')
        axs[0,1].hist(cmat[:,1],bins=100)
        axs[0,1].set_title('Centered 1 histogram')
        axs[1,0].hist(cmat[:,2],bins=100)
        axs[1,0].set_title('Centered 2 histogram')
        axs[1,0].set_ylabel('# images')
        axs[1,0].set_xlabel('Pixel value')
        axs[1,1].hist(cmat[:,3],bins=100)
        axs[1,1].set_title('Centered 3 histogram')
        axs[1,1].set_xlabel('Pixel value')
        PlotWindow(fig)
    else:
        ctrain = matrix_and_vector_to_dataset(solutions['ctrainmatrix'], solutions['ctrainlabelvec'])
        cdev = matrix_and_vector_to_dataset(solutions['cdevmatrix'], solutions['cdevlabelvec'])
        ctest = matrix_and_vector_to_dataset(solutions['ctestmatrix'], solutions['ctestlabelvec'])


    ###############################################################
    # Step 3a: Compute the principal components transform, and show PC
    if args.steps is None or '3' in args.steps:
        transform, variances = todo_find_transform(ctrain)
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(2,2,sharex=True,sharey=True)
        im = vector_to_image(transform[:,0])
        im = (im-np.amin(im))/(np.amax(im)-np.amin(im)+1e-6)
        axs[0,0].clear()
        axs[0,0].imshow(im)
        axs[0,0].set_title('PCA0 as image')
        im = vector_to_image(transform[:,1])
        im = (im-np.amin(im))/(np.amax(im)-np.amin(im)+1e-6)
        axs[0,1].clear()
        axs[0,1].imshow(im)
        axs[0,1].set_title('PCA1 as image')
        im = vector_to_image(transform[:,2])
        im = (im-np.amin(im))/(np.amax(im)-np.amin(im)+1e-6)
        axs[1,0].clear()
        axs[1,0].imshow(im)
        axs[1,0].set_title('PCA2 as image')
        im = vector_to_image(transform[:,3])
        im = (im-np.amin(im))/(np.amax(im)-np.amin(im)+1e-6)
        axs[1,1].clear()
        axs[1,1].imshow(im)
        axs[1,1].set_title('PCA3 as image')
        PlotWindow(fig)

        # Step 3b: show the cumulative variances
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        ax = fig.subplots()
        ax.clear()
        ax.plot(100*np.cumsum(variances)/np.sum(variances))
        ax.set_title('Percent of variance explained vs. # PC dimensions')
        PlotWindow(fig)
    else:
        transform = solutions['transform']
        variances = solutions['variances']

    ###############################################################
    # Step 4: Transform all three centered datasets
    if args.steps is None or '4' in args.steps:
        ttrain, tdev, ttest = todo_transform_datasets(ctrain, cdev, ctest, transform)
        dt = len(ttrain[0].x)
        dc = len(ctrain[0].x)
        ndev = len(cdev)
        print('Ndev=%d, dt=%d, dc=%d'%(ndev,dt,dc))
        tmat = dataset_to_matrix(ttrain)
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(2,2,sharex=True)
        axs[0,0].clear()
        axs[0,0].plot(tmat[:,0],tmat[:,1],'.')
        axs[0,1].clear()
        axs[0,0].set_title('PCA1 vs. PCA0')
        axs[0,1].plot(tmat[:,0],tmat[:,2],'.')
        axs[0,1].set_title('PCA2 vs. PCA0')
        axs[1,0].clear()
        axs[1,0].plot(tmat[:,0],tmat[:,3],'.')
        axs[1,0].set_title('PCA3 vs. PCA0')
        axs[1,1].clear()
        axs[1,1].plot(tmat[:,0],tmat[:,4],'.')
        axs[1,1].set_title('PCA4 vs. PCA0')
        PlotWindow(fig)

        # Step 4b: show histograms of transformed train set
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(2,2,sharex=True)
        axs[0,0].hist(tmat[:,0],bins=100)
        axs[0,0].set_title('PCA0 histogram')
        axs[0,0].set_ylabel('# images')
        axs[0,1].hist(tmat[:,1],bins=100)
        axs[0,1].set_title('PCA1 histogram')
        axs[1,0].hist(tmat[:,2],bins=100)
        axs[1,0].set_title('PCA2 histogram')
        axs[1,0].set_ylabel('# images')
        axs[1,0].set_xlabel('Feature value')
        axs[1,1].hist(tmat[:,3],bins=100)
        axs[1,1].set_title('PCA3 histogram')
        axs[1,1].set_xlabel('Feature value')
        PlotWindow(fig)
    else:
        ttrain = matrix_and_vector_to_dataset(solutions['ttrainmatrix'], solutions['ttrainlabelvec'])
        tdev = matrix_and_vector_to_dataset(solutions['tdevmatrix'], solutions['tdevlabelvec'])
        ttest = matrix_and_vector_to_dataset(solutions['ttestmatrix'], solutions['ttestlabelvec'])


    ###############################################################
    # Step 5: Find and plot the distances matrix with full vector size
    if args.steps is None or '5' in args.steps:
        fullsize = len(ttrain)
        Dfull = todo_distances(ttrain, tdev, fullsize)
    else:
        Dfull = solutions['Dfull']

    if args.steps is None or '6' in args.steps:
        hypsfull = todo_nearest_neighbor(ttrain, Dfull)
    else:
        hypsfull = solutions['hypsfull']

    if args.steps is None or '7' in args.steps:
        refs = [ datum.y  for datum in tdev ]
        accuracyfull, confusionfull = todo_compute_accuracy(tdev, hypsfull)
        print('w/feature size=%d, dev set accuracy=%d%%'%(fullsize,int(100*accuracyfull)))
        print('Classes are: %s'%(str(personnames)))
        print('Confusion matrix is:')
        print(confusionfull)
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        ax = fig.subplots()
        im = (Dfull-np.amin(Dfull))/(np.amax(Dfull)-np.amin(Dfull)+1e-6)
        ax.clear()
        ax.imshow(im)
        ax.set_title('distances from train to dev data, %d dims, acc=%d%%'%(fullsize,int(100*accuracyfull)))
        ax.set_ylabel('train datum')
        ax.set_xlabel('dev datum')
        PlotWindow(fig)
    else:
        confusionfull = solutions['confusionfull']

    ###############################################################
    # Step 6: Find the best feature size using devset
    if args.steps is  None or '8' in args.steps:
        bestsize, accuracies = todo_find_bestsize(ttrain, tdev, variances)
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        ax = fig.subplots()
        ax.plot(accuracies)
        ax.set_title('Dev Accuracy vs. Feature Dim: Best Size=%d'%(bestsize))
        ax.set_xlabel('Number of PCA Dimensions')
        ax.set_ylabel('Dev Set Accuracy')
        PlotWindow(fig)
    else:
        accuracies = solutions['accuracies']
        bestsize = np.argmax(accuracies)

    ###############################################################
    # Step 7: Show the test confusion matrix
    if args.steps is None or '5' in args.steps:
        Dtest = todo_distances(ttrain, ttest, bestsize)
    else:
        Dtest = solutions['Dtest']

    if args.steps is None or '6' in args.steps:
        hypstest = todo_nearest_neighbor(ttrain, Dtest)
    else:
        hypstest = solutions['hypstest']

    if args.steps is None or '7' in args.steps:
        accuracytest, confusiontest = todo_compute_accuracy(ttest, hypstest)
        print('w/feature size=%d, test set accuracy=%d%%'%(bestsize,int(100*accuracytest)))
        print('Classes are: %s'%(str(personnames)))
        print('Confusion matrix is:')
        print(confusiontest)
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        ax = fig.subplots()
        im = (Dtest-np.amin(Dtest))/(np.amax(Dtest)-np.amin(Dtest)+1e-6)
        ax.clear()
        ax.imshow(im)
        ax.set_title('distances from train to test data, %d dims, acc=%d%%'%(bestsize,int(100*accuracytest)))
        ax.set_ylabel('train datum')
        ax.set_xlabel('test datum')
        PlotWindow(fig)
    else:
        confusiontest = solutions['confusiontest']

    ###############################################################
    # Now create an hdf5 file with your results
    with h5py.File('results.hdf5', 'w') as f:
        f.create_dataset('trainmatrix', data=dataset_to_matrix(train))
        f.create_dataset('trainlabelvec', data=dataset_to_labelvec(train))
        f.create_dataset('devmatrix', data=dataset_to_matrix(dev))
        f.create_dataset('devlabelvec', data=dataset_to_labelvec(dev))
        f.create_dataset('testmatrix', data=dataset_to_matrix(test))
        f.create_dataset('testlabelvec', data=dataset_to_labelvec(test))
        f.create_dataset('mu',data=mu)
        f.create_dataset('ctrainmatrix', data=dataset_to_matrix(ctrain))
        f.create_dataset('ctrainlabelvec', data=dataset_to_labelvec(ctrain))
        f.create_dataset('cdevmatrix', data=dataset_to_matrix(cdev))
        f.create_dataset('cdevlabelvec', data=dataset_to_labelvec(cdev))
        f.create_dataset('ctestmatrix', data=dataset_to_matrix(ctest))
        f.create_dataset('ctestlabelvec', data=dataset_to_labelvec(ctest))
        f.create_dataset('transform', data=transform)
        f.create_dataset('variances', data=variances)
        f.create_dataset('ttrainmatrix', data=dataset_to_matrix(ttrain))
        f.create_dataset('ttrainlabelvec', data=dataset_to_labelvec(ttrain))
        f.create_dataset('tdevmatrix', data=dataset_to_matrix(tdev))
        f.create_dataset('tdevlabelvec', data=dataset_to_labelvec(tdev))
        f.create_dataset('ttestmatrix', data=dataset_to_matrix(ttest))
        f.create_dataset('ttestlabelvec', data=dataset_to_labelvec(ttest))
        f.create_dataset('Dfull', data=Dfull)
        f.create_dataset('hypsfull', data=hypsfull)
        f.create_dataset('confusionfull', data=confusionfull)
        f.create_dataset('accuracies', data=accuracies)
        f.create_dataset('Dtest', data=Dtest)
        f.create_dataset('hypstest', data=hypstest)
        f.create_dataset('confusiontest', data=confusiontest)

    print('Done!  Now try python run_tests.py.')
