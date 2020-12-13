import os, h5py, glob, re, argparse
from PIL import Image
import numpy as  np

###############################################################################
# Utility functions: IoU, rect_regression, initialize_weights, load_weights
def IoU(rect1, rect2):
    '''Calculate intersection-over-union between two rects, or a  rect and a set  of rects'''
    ileft = np.maximum(rect1[0],rect2[0])
    iright = np.minimum(rect1[0]+rect1[2],rect2[0]+rect2[2])
    itop = np.maximum(rect1[1],rect2[1])
    ibottom = np.minimum(rect1[1]+rect1[3],rect2[1]+rect2[3])
    intersection = (iright-ileft)*(ibottom-itop)
    union = rect1[2]*rect1[3] + rect2[2]*rect2[3] - intersection
    return(intersection/union)

def rect_regression(rect, anchor):
    '''Convert a rectangle into a regression target, with respect to  a given anchor rect'''
    return(np.array([(rect[0]-anchor[0])/anchor[2],(rect[1]-anchor[1])/anchor[3],
                     np.log(rect[2]/anchor[2]),np.log(rect[3]/anchor[3])]))

def initialize_weights():
    '''Randomly initialize weight tensors'''
    W1 = 0.001*np.random.randn(4608,512)
    W2 = 0.001*np.random.randn(9,512,5)
    return(W1,W2)

def load_weights(filename):
    '''Load pre-trained weights from an HDF5 file'''
    with h5py.File(filename,'r')  as f:
        W1 = f['W1'][:]
        W2 = f['W2'][:]
    return(W1,W2)

def save_weights(filename,W1,W2):
    '''Save trained weights'''
    with h5py.File(filename,'w') as f:
        f.create_dataset('W1',data=W1)
        f.create_dataset('W2',data=W2)

###############################################################################
# Use:
# mp3_dataset = MP3_Dataset()
# for i in range(num_iters):
#    datum = mp3_dataset[i % len(mp3_dataset)]
#    hypothesis, hidden = do_a_forward_pass(datum['features'])
#    backprop = do_a_backward_pass(hypothesis, hidden, datum['target'])
#    update_weights(hypothesis, hidden, backprop, weights)
#
class MP3_Dataset(object):
    def __init__(self, datadir):
        '''
        Initialize: 
        read the list self.imagefiles from datadir/images
        generate self.anchors using the Faster RCNN configuration, resized for 224-pix images
        '''
        self.imagefiles = glob.glob(os.path.join(datadir, 'images/*/*.jpg'))
        sizes = np.array([128,256,512])*(224/1024) # image had 1024 rows; resized has 224
        w = np.outer(sizes, np.sqrt([0.5,1,2])).flatten()
        h = np.outer(sizes, np.sqrt([2,1,0.5])).flatten()
        # anchor rectangles: [x,y,w,h], where x, y, w, h are each 196x9 matrices (xy by a)
        self.anchors = np.array([
            [[x for a in range(9)] for y in np.arange(8,16*14,16) for x in np.arange(8,16*14,16)],
            [[y for a in range(9)] for y in np.arange(8,16*14,16) for x in np.arange(8,16*14,16)],
            [[w[a] for a in range(9)] for y in np.arange(8,16*14,16) for x in np.arange(8,16*14,16)],
            [[h[a] for a in range(9)] for y in np.arange(8,16*14,16) for x in np.arange(8,16*14,16)]])
        
    def __len__(self):
        return(len(self.imagefiles))
    
    def target_tensor(self, rects):
        '''
        Convert a set of rects into a 9x196x5 Faster-RCNN training target, 
        using self.anchors as the anchor rectangles,
        and using the function  rect_regression to generate each target individually.
        '''
        target = np.zeros((9,196,5))
        for rect in rects:
            similarities = IoU(rect, self.anchors)
            xylist,alist = (similarities > 0.7).nonzero()
            for xy,a in zip(xylist,alist):  
                target[a,xy,0:4] = rect_regression(rect,self.anchors[:,xy,a])
                target[a,xy,4] = 1
            else:
                xy,a  = np.unravel_index(np.argmax(similarities), similarities.shape)
                target[a,xy,0:4] = rect_regression(rect,self.anchors[:,xy,a])
                target[a,xy,4] = 1
        return(target)
    
    def __getitem__(self, n):
        '''
        The n'th datum, from the dataset, includes:
        datum['image'] = the raw image
        datum['features'] [1x512x14x14] = the image features, pre-computed by a VGG16 network,
          normalized to unit-L1 norm in order to keep the gradients to a reasonable scale
        datum['rects'] = the raw rectangles,
          converted from WIDER format (xmin,ymin,w,h) to Faster RCNN format (xmid,ymid,w,h).
        datum['target'] [9x196x5] = 
          datum['target'][a,i,0:4] = regression target for anchor a at position i
          datum['target'][a,i,4] = classification target for anchor a at position i
        '''
        imagepath = self.imagefiles[n]
        image = np.asarray(Image.open(imagepath)).astype('float64')
        image = (image - np.amin(image)) / (1e-6+np.amax(image)-np.amin(image))
        with h5py.File(re.sub(r'images','features',re.sub(r'.jpg','.hdf5',imagepath)),'r') as f:
            features = f['features'][:] / np.sum(np.abs(f['features'][:]))
        with open(re.sub(r'images','rects',re.sub(r'.jpg','.txt',imagepath))) as f:
            rects=np.array([[float(w) for w in line.strip().split() ] for line in f.readlines()])
        for rect in rects:
            rect[0] += 0.5*rect[2]
            rect[1] += 0.5*rect[3]
        target = self.target_tensor(rects)
        return({'image':image, 'features':features, 'rects':rects, 'target':target})

###############################################################################
# TODO: here are the functions that you need to write

def todo_concatenate(features):
    '''
    Input: 
    features [1x512x14x14] - These are the activations from the last convolutional layer 
      in a standard VGG16 package.
      This layer has 512 channels, with a 14x14 image (y by x) in each channel.
      You should treat it as a 512d feature vector, in each of the 14x14=196 locations.
    Output: 
    concatenation [196x4608] - 4608d vector for each of the 14x14=196 reference points.
      The 196 vectors should be arranged in the usual numpy.unravel_index(order='C') order:
      the last coordinate of features (x) changes most quickly, the second-to-last (y) more slowly.
      Each 4608d vector is the concatenation of nine neighboring 512d vectors.
      These are the vectors that were stored in the features matrix at locations
      (y-1,x-1), (y-1,x), (y-1,x+1), (y,x-1), ..., (y+1, x+1).
    '''
    concatenation = np.zeros((196,4608))
    limit = [-1, 0 ,1]
    m = 0
    n = 0
    for i in range(14):
        for j in range(14):
            idx = 0
            for offset_x in limit:
                for offset_y in limit:
                    m = i+offset_x
                    n = j+offset_y
                    m = np.max((0, m))
                    n = np.max((0, n))
                    m = np.min((13,m))
                    n = np.min((13,n))
                    concatenation[i*14+j, 512*idx:512*(idx+1)] = features[0,:,m,n]
                    idx = idx + 1
    return(concatenation)

def sigmoid(excitation):
    '''
    You should know how to write your own sigmoid function.
    What you might not know is that np.exp(-x) generates NaN if x<-100 or so.
    In order to get around that problem, this function just leaves activation=0
    if excitation <= -100.  Feel free to use this, to avoid NaNs.
    '''
    activation = np.zeros(excitation.shape)
    activation[excitation > -100] = 1/(1+np.exp(-excitation[excitation > -100]))
    return(activation)

def todo_forward(concatenation, W1, W2):
    '''
    Input: 
    concatenation [196x4608] - two-layer network inputs
    W1 [4608x512] - layer 1 weights
    W2 [9x512x5] - layer 2 weights, for each of the 9 different anchors at each position
    Output:
    hidden [196x512] = hidden layer  activations
      activation = layernorm(ReLU(excitation))
    hypothesis [9x196x5] - hypotheses for each of the 9 anchors for each of 196 positions
      hypothesis[a,i,4] = classification hypothesis, a number between 0 and 1 (sigmoid activation)
      hypothesis[a,i,0:4] = regression hypothesis, unbounded real numbers (linear activation)
    '''
    hidden = np.matmul(concatenation, W1)
    hidden = np.multiply(hidden, (hidden > 0))
    hypothesis = np.matmul(hidden, W2)
    hypothesis[:,:,4] = sigmoid(hypothesis[:,:,4])
    return(hypothesis, hidden)

def todo_detect_rectangles(hypothesis, number_to_return, mp3_dataset):
    '''
    Input:
    hypothesis [9x196x5] - neural net outputs for each of 9 anchors at each of 196 positions
    number_to_return [scalar int] - the number of rectangles to return
    mp3_dataset [MP3_Dataset object] - a dataset containing mp3_dataset.anchors
    Output:
    best_rects [number_to_return x 4] - [x,y,w,h] rectangles most likely to contain faces.
      You should find the number_to_return hypotheses (a,i) 
      with the highest values of hypothesis[a,i,4],
      then convert their corresponding hypothesis[a,i,0:4] 
      from regression targets back into rectangles
      (i.e., reverse the process in rect_regression()).
    '''

    idx = np.empty(number_to_return, dtype=np.int)
    for i in range(number_to_return):
        ith = np.argpartition(hypothesis[:,:,4], (9*196-1-i), axis=None)
        ith = ith[9*196-1-i]
        #print((np.argpartition(hypothesis[:,:,4], (9*196-1-i), axis=None)).())
        idx[i] = ith
        #print(idx[i])
    best_rects = hypothesis[:, :, 0:4].reshape((1764, 4))[idx, :]
    anchor = mp3_dataset.anchors.T
    anchor = anchor.reshape((1764, 4))
    anchor = anchor[idx, :]
    best_rects[:,0:2] = anchor[:, 0:2]+best_rects[:, 0:2]*anchor[:, 2:4]
    best_rects[:,2:4] = np.exp(best_rects[:, 2:4])
    best_rects[:,2:4] = best_rects[:,2:4]*anchor[:,2:4]
    return(best_rects)

def todo_outputgrad(hypothesis, target):
    '''
    Inputs:
    hypothesis [9x196x5] - neural net outputs for each of 9 anchors at each of 196 positions
    target [9x196x5] - training targets at each of 9 anchors at each of 196 positions
    Output:
    outputgrad [9x196x5] - derivative of loss w.r.t. the _excitations_ in the output layer.
      loss = average over a (9 of them), of the average over i (196 of them), of the 
        binary cross entropy loss comparing hypothesis[a,i,4] to target[a,i,4], plus
        0.5 times the mean-squared-error comparing hypothesis[a,i,0:4] to target[a,i,0:4],
        where the mean is computed over all (a,i) pairs.
    Note: you don't have to compute the loss; you just need to compute 
    outputgrad = the derivative of the loss w.r.t. the excitations in the output layer.
    '''
    #outputgrad = np.zeros((9,196,5))
    #epsilon = np.zeros((9,196,5))
    hypothesis_array = np.array(hypothesis)
    target_array = np.array(target)
    epsilon = 1/(1764) * (hypothesis_array - target_array)
    outputgrad = epsilon
    target_reshape = target[:,:,4].reshape((9, 196, 1))
    outputgrad[:,:,0:4] = outputgrad[:,:,0:4] * target_reshape
    return(outputgrad)

def todo_backprop(outputgrad, hidden, W2):
    '''
    Inputs:
    outputgrad  [9x196x5] = derivative of loss w.r.t. output layer excitations.
    hidden [196x512] = hidden layer activations
    W2 [9x512x5] = second layer weights
    Outputs:
    backprop [196x512] = derivative of loss w.r.t. hidden layer excitations.
    '''
    backprop = np.zeros((196,512))
    for i in range(9):
        backprop += np.matmul(outputgrad[i], W2[i].T)
    backprop = backprop * (hidden > 0)
    return(backprop)

def todo_weightgrad(outputgrad, backprop, hidden, concatenation):
    '''
    Inputs:
    outputgrad  [9x196x5] = derivative of loss w.r.t. output layer excitations.
    backprop [196x512] = derivative of loss w.r.t. hidden layer excitations.
    hidden [196x512] = hidden layer activations
    concatenation [196x4608] - inputs to the two-layer network.
    Outputs:
    dW1 [4608x512] = derivative of loss w.r.t. each element of W1.
    dW2 [9x512x5] = derivative of loss w.r.t. each element of W2.
    '''
    concatenation_T = concatenation.T
    dW1 = np.matmul(concatenation_T, backprop)
    hidden_T = hidden.T
    dW2 = np.matmul(hidden_T, outputgrad)
    return(dW1, dW2)

def todo_update_weights(W1, W2, dW1, dW2, learning_rate):
    '''
    Input: 
    W1 [4608x512] = first layer weights
    W2 [9x512x5] = second layer weights
    dW1 [4608x512] = derivative of loss w.r.t. each element of W1
    dW2 [9x512x5] = derivative of loss w.r.t. each element of W2
    learning_rate = scalar learning rate
    Output:
    new_W1 [4608x512] = new W1, after one step of stochastic gradient descent.
    new_W2 [9x512x5] = new W2, after one step of stochastic gradient descent.
    '''
    
    new_W1 = W1 - dW1*learning_rate
    new_W2 = W2 - dW2*learning_rate
    return(new_W1, new_W2)


###############################################################################
if __name__=="__main__":
    parser = argparse.ArgumentParser('Run MP3 to generate results.hdf5.')
    parser.add_argument('--datadir',default='data',help='''Set datadir.  Default: "data"''')
    parser.add_argument('-w','--weights',
                        help='''Name of HDF5 file containing initial weights.
                        Default: weights_trained.hdf5''')
    parser.add_argument('-i','--iters',metavar='N',type=int,default=1,
                        help='''# of training iterations, with batch size=1 image''')
    args = parser.parse_args()

    # Load the weights
    if args.weights == None:
        args.weights = 'weights_trained.hdf5'
    with h5py.File(args.weights,'r') as f:
        W1 = f['W1'][:]
        W2 = f['W2'][:]

    # Load the data
    mp3_dataset = MP3_Dataset(args.datadir)
    
    # Perform the training iterations
    for i in range(args.iters):
        if i % len(mp3_dataset) == 0:
            print(i)
        else:
            print('.',end='')
        datum = mp3_dataset[i % len(mp3_dataset)]
        concatenation = todo_concatenate(datum['features'])
        hypothesis, hidden = todo_forward(concatenation, W1, W2)
        outputgrad = todo_outputgrad(hypothesis, datum['target'])
        backprop = todo_backprop(outputgrad, hidden, W2)
        dW1, dW2 = todo_weightgrad(outputgrad, backprop, hidden, concatenation)
        W1, W2 = todo_update_weights(W1, W2, dW1, dW2, 0.01)

    # Test
    best_rects = todo_detect_rectangles(hypothesis, 10, mp3_dataset)

    # Save results
    datadir_name = args.datadir
    weights_name = os.path.splitext(args.weights)[0]
    experiment_name = '%s_%s_%d'%(datadir_name,weights_name,args.iters)
    with h5py.File('results_'+experiment_name+'.hdf5','w')  as f:
        f.create_dataset('features',data=datum['features'])
        f.create_dataset('target',data=datum['target'])
        f.create_dataset('concatenation',data=concatenation)
        f.create_dataset('hypothesis',data=hypothesis)
        f.create_dataset('hidden',data=hidden)
        f.create_dataset('outputgrad',data=outputgrad)
        f.create_dataset('backprop',data=backprop)
        f.create_dataset('dW1',data=dW1)
        f.create_dataset('dW2',data=dW2)
        f.create_dataset('best_rects',data=best_rects)
        f.create_dataset('W1',data=W1)
        f.create_dataset('W2',data=W2)
    with h5py.File('weights_'+experiment_name+'.hdf5','w') as f:
        f.create_dataset('W1',data=W1)
        f.create_dataset('W2',data=W2)
        
