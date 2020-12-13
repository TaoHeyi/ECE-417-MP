import os, h5py, wave
import numpy as np

###############################################################################
# First, some utility functions to help you load and convert image data
class Datum(object):
    def __init__(self, datumpath=None, label=None, labels=None, x=None, y=None, fs=None):
        '''Load self.x=vectorized image, self.y=integer index of personname'''
        if datumpath is not None:
            w = wave.open(datumpath,'rb')
            fs = w.getframerate()                                                                   #return sampling frequency
            nsamples = w.getnframes()                                                               #return number of audio frames
                                                                                                    #read nsamples frame first,
                                                                                                    #Interpret a buffer as a 1-dimensional array
            self.x = np.frombuffer(w.readframes(nsamples),dtype=np.int16).astype('float32')/32768
                                                                                                    #np.amax returns the maximum of an array or maximum along an axis.

            self.x = self.x / np.amax(np.abs(self.x))
            centroid = int(np.average(np.where(np.absolute(self.x)>0.5)))                           #np.where returns where the condition is true as an array

            s = max(0,min(len(self.x)-int(0.5*fs),centroid-int(0.25*fs)))
            e = max(int(0.5*fs),min(len(self.x),centroid+int(0.25*fs)))
            self.x = self.x[s:e]
        if label is not None and labels is not None:
            self.y = labels.index(label)
        if x is not None:
            self.x = x.copy()
        if y is not None:
            self.y = y
        if fs is not None:
            self.fs = fs

def load_datasets(datapath):
    '''Load image datasets; divide into train, dev, and test subsets'''
    train, dev, test = [], [], []
    labels = os.listdir(datapath)
    n = 0
    for label in labels:
        for filename in os.listdir(os.path.join(datapath,label)):
            datumpath = os.path.join(datapath,label,filename)
            if n % 5 == 4:
                test.append(Datum(datumpath, label, labels))
            elif n % 5 == 3:
                dev.append(Datum(datumpath, label, labels))
            else:
                train.append(Datum(datumpath, label, labels))
            n += 1
    return(train,dev,test,labels)

def dataset_to_matrix(dataset):
    '''Create a matrix in which each row is a vectorized observation from the dataset'''
    return(np.array([datum.x for datum in dataset]))

###############################################################################
# TODO: here are the functions that you need to write
def todo_spectrograms(nwtrain, nwdev, nwtest, fs, size, skip, fftlen):
    '''
    Compute spectrogram: sg[i,:,t] = log(1e-7 + abs(fft.rfft(nw[i,samplerange], fftlen))
    The 1e-7 is to make sure you don't take the log of zero.
    The samplerange extends from t*skip  seconds to t*skip+size seconds.
    With 0.5 seconds of input, you'll have 1+int((0.5-size)/skip) frames.
    '''
    nfreqs = int(fftlen/2+1)
    nframes = 1+int((0.5-size)/skip)

    sgtrain = np.zeros((nwtrain.shape[0], nfreqs, nframes))
    for i in range(nwtrain.shape[0]):
        for t in range(nframes):
            sgtrain[i, :, t] = np.log(1e-7 + abs(np.fft.rfft(nwtrain[i, int(fs*t*skip):int(fs*(t*skip+size))], fftlen)))

    sgdev = np.zeros((nwdev.shape[0], nfreqs, nframes))
    for i in range(nwdev.shape[0]):
        for t in range(nframes):
            sgdev[i, :, t] = np.log(1e-7 + abs(np.fft.rfft(nwdev[i, int(fs * t * skip):int(fs*(t*skip + size))], fftlen)))

    sgtest = np.zeros((nwtest.shape[0], nfreqs, nframes))
    for i in range(nwtest.shape[0]):
        for t in range(nframes):
            sgtest[i, :, t] = np.log(1e-7 + abs(np.fft.rfft(nwtest[i, int(fs * t * skip):int(fs*(t*skip + size))], fftlen)))

    return(sgtrain,sgdev,sgtest)

def todo_melfilters(nfreqs, nfilts, fs):
    '''
    Return filters, centers.
    centers = (nfilts+2) center frequencies, in Hertz.
      These should be uniformly spaced on the mel-scale, so that
      centers[0] = 0
      centers[-1] = fs/2
      mel(centers[m]) - mel(centers[m-1]) is a constant for all m,
      where
      mel(f) = 2595*log(1+(f/700)).
    filters: an nfilts by nfreqs array, such that
      filters[m,dftbin(m)] = 0
      filters[m,dftbin(m+1)] = 1
      filters[m,dftbin(m+2)] = 0
      and intervening frequency samples are linearly interpolated between these.
      dftbin(m) is the dft bin corresponding to Hertz frequency centers[m]
    '''
    def hz2mel(f):
        return(2595*np.log(1+(f/700)))
    def mel2hz(mel):
        return(700*(np.exp(mel/2595)-1))

    hz_max = fs/2
    mel_max = hz2mel(hz_max)
    mel_step = mel_max / (nfilts + 1)
    freq_step = hz_max/ (nfreqs - 1)
    filters = np.zeros((nfilts, nfreqs))
    centers = np.array([mel2hz(i*mel_step) for i in range (nfilts+2)])

    for i in range(nfilts):
        start = int(centers[i]/freq_step)
        middle = int(centers[i+1]/freq_step)
        end = int(centers[i+2]/freq_step)
        first_half_slope = 1/(middle-start)
        second_half_slope = 1/(middle-end)

        for j in range(start, end):
            if j < middle:
                filters[i, j] = (j-start)*first_half_slope
            else:
                filters[i, j] = 1+(j-middle)*second_half_slope

    return(filters, centers)

def todo_filterbank(sgtrain, sgdev, sgtest, melfilters):
    '''
    Compute mel filterbank coefficients for the train, dev, and test corpora.
    First, exponentiate the spectrogram, to get the magnitude fft.
    Second, multiply the magnitude fft by the melfilters, to get an (nfilts x nframes) array.
    Third, take log(1e-7+result) again, to get back to log amplitudes.
    '''
    nfilts = melfilters.shape[0]
    fbtrain = np.zeros((sgtrain.shape[0], nfilts, sgtrain.shape[2]))
    fbdev = np.zeros((sgdev.shape[0], nfilts, sgdev.shape[2]))
    fbtest = np.zeros((sgtest.shape[0], nfilts, sgtest.shape[2]))

    train_temp = np.exp(sgtrain)
    dev_temp = np.exp(sgdev)
    test_temp = np.exp(sgtest)

    for i in range(train_temp.shape[0]):
        fbtrain[i] = np.dot(melfilters, train_temp[i])
    for i in range(dev_temp.shape[0]):
        fbdev[i] = np.dot(melfilters, dev_temp[i])
    for i in range(test_temp.shape[0]):
        fbtest[i] = np.dot(melfilters, test_temp[i])

    fbtrain_final = np.log(1e-7+fbtrain)
    fbdev_final = np.log(1e-7+fbdev)
    fbtest_final = np.log(1e-7+fbtest)
    return(fbtrain_final, fbdev_final, fbtest_final)

def todo_gtfilters(nfilts, fs, duration):
    '''
    Create the impulse responses of gammatone filters, with length=int(fs*duration) samples.
    Return filters, centers.
    centers = (nfilts+2) center frequencies, in Hertz.
      These should be uniformly spaced on the ERB-scale (ERBS), so that
      centers[0] = 0
      centers[-1] = fs/2
      hz2erbs(centers[m]) - hz2erbs(centers[m-1]) is a constant for all m,
      where
      hz2erbs(f) = 11.17268*log(1+46.06538*f/(f+14678.49))
    filters: an nfilts by int(duration*fs) array, such that
      filters[m,:] is the impulse response of the gammatone filter
      centered at f=centers[m+1], and with bandwith b=erb(centers[m+1]),
      where erb_at_f(f) = derbs/df = 6.23*(f/1000)^2 + 93.39*(f/1000) + 28.52.
      The impulse response of a gammatone is
      (t*fs)^3 * exp(-2*pi*b*t) * cos(2*pi*f*t), for t in seconds,
      renormalized so that the L2-norm of each impulse response is 1.
    '''
    def hz2erbs(f):
        return(11.17268*np.log(1+46.06538*f/(f+14678.49)))
    def erbs2hz(erbs):
        return(676170.4/(47.06538-np.exp(0.08950404*erbs))-14678.49)
    def erb_at_hz(f):
        return(6.23*np.square(f/1000) + 93.39*(f/1000) + 28.52)

    t = np.linspace(0,duration, int(fs*duration), endpoint=False)
    gtfilters = np.zeros((nfilts, int(fs*duration)))

    hz_max = fs / 2
    erb_max = hz2erbs(hz_max)
    erb_step = erb_max / (nfilts + 1)
    centers = np.array([erbs2hz(i * erb_step) for i in range(nfilts + 2)])

    for i in range(nfilts):
        f = centers[i+1]
        b = erb_at_hz(f)
        for j in range(len(t)):
            time = t[j]
            gtfilters[i, j] = (time*fs)**3 * np.exp(-2*np.pi*b*time) * np.cos(2*np.pi*f*time)

    for i in range(gtfilters.shape[0]):
        gtfilters[i] = gtfilters[i]/np.linalg.norm(gtfilters[i])

    return(gtfilters, centers)

def todo_gammatone(nwtrain, nwdev, nwtest, gtfilters, fs, skip):
    '''
    Compute gammatone coefficients from the nwtrain, nwdev, and nwtest waveforms.
    Convolve each waveform with each of the gammatone filters, using mode='valid'
    so that no zero-padding is used.
    Add the squared output samples, in non-overlapping frames of skip seconds each,
    then take log(1e-7 + sum-square-result).
    For 0.5 second waveforms, that should give you output matrices of size nfilts by nframes,
    where nfilts is the number of gtfilters, and nframes is int((len(signal)-len(filter)+1)/(fs*skip)).
    '''
    lensignal = nwtrain.shape[1]
    nfilts, lenfilter = gtfilters.shape
    nframes = int((lensignal-lenfilter+1)/(fs*skip))
    gttrain = np.zeros((nwtrain.shape[0],nfilts,nframes))
    gtdev = np.zeros((nwdev.shape[0],nfilts,nframes))
    gttest = np.zeros((nwtest.shape[0],nfilts, nframes))


    for i in range(nwtrain.shape[0]):
        for j in range(nfilts):
            convolve_result = np.convolve(nwtrain[i], gtfilters[j], mode='valid')

            for k in range(nframes):
                temp = 0
                #temp = np.linalg.norm(convolve_result[int(k*skip*fs):int((k+1)*skip*fs)])
                for l in range(int(k*skip*fs),int((k+1)*skip*fs)):
                    temp += convolve_result[l]**2
                #square_temp = temp**2
                gttrain[i,j,k] = np.log(1e-7 + temp)

    for i in range(nwdev.shape[0]):
        for j in range(nfilts):
            convolve_result = np.convolve(nwdev[i], gtfilters[j], mode='valid')

            for k in range(nframes):
                temp = 0
                #temp = np.linalg.norm(convolve_result[int(k * skip * fs):int((k + 1) * skip * fs)])
                for l in range(int(k*skip*fs),int((k+1)*skip*fs)):
                    temp += convolve_result[l]**2
                #square_temp = temp ** 2
                gtdev[i, j, k] = np.log(1e-7 + temp)

    for i in range(nwtest.shape[0]):
        for j in range(nfilts):
            convolve_result = np.convolve(nwtest[i], gtfilters[j], mode='valid')

            for k in range(nframes):
                temp1 = 0
                #temp = np.linalg.norm(convolve_result[int(k * skip * fs):int((k + 1) * skip * fs)])
                #print(convolve_result[int(k * skip * fs):int((k + 1) * skip * fs)])
                #print(temp)
                for l in range(int(k*skip*fs),int((k+1)*skip*fs)):
                    temp1 += convolve_result[l]**2
                    #print(convolve_result[l])
                    #print(temp1)
                #print(temp1)
                #return(gttrain,gtdev,gttest)
                #print(temp)
                #print(temp1)
                gttest[i, j, k] = np.log(1e-7 + temp1)
    """
    for i in range(nwtrain.shape[0]):
        for j in range(nfilts):
            samples = np.convolve(nwtrain[i], gtfilters[j], mode='valid')
            for k in range(nframes):
                sum_square_result = np.square(np.linalg.norm(samples[int(k * skip * fs):int((k + 1) * skip * fs)]))
                gttrain[i, j, k] = np.log(1e-7 + sum_square_result)
    """
    return(gttrain,gtdev,gttest)

def todo_distances(train, dev):
    '''
    Return a matrix dist such that dist[i,j]=distance(train[i,:,:],dev[j,:,:]).
    This function differs from MP1 only in that the inputs are 3D tensors, not datasets.
    '''
    D = np.zeros(shape=(train.shape[0], dev.shape[0]))
    for i in range(train.shape[0]):
        for j in range(dev.shape[0]):
            D[i, j] = np.linalg.norm(train[i] - dev[j])
    return (D)

def todo_nearest_neighbor(train, D):
    '''
    Given the dataset train, and the (NTRAINxNTEST) matrix dist, returns
    an int numpy array of length NTEST, specifying the person number (y) of the training token
    that is closest to each of the NTEST test tokens.
    For this function, you should be able to copy in your code from MP1, without change.
    '''
    #(ntrain, ndev) = D.shape
    #hyps = np.zeros(ndev, dtype='int')
    #raise NotImplementedError('You need to write this part!')
    closest = np.argmin(D, axis=0)
    hyps = np.array([train[index].y for index in closest])
    # print(hyps)
    return (hyps)

def todo_compute_accuracy(test, hyps):
    '''
    Compute accuracy, confusion.
    confusion is a (4x4) matrix.
    confusion[ref,hyp] is the number of test tokens for which test[i].y==ref
    but hyps[i]==hyp.
    accuracy is a scalar between 0 and 1, equal to the total fraction of
    hyps that are correct.
    For this function, you should be able to copy in your code from MP1, without change.
    '''
    confusion = np.zeros((4, 4))
    total = len(hyps)
    for i in range(len(hyps)):
        ref = test[i].y
        hyp = hyps[i]
        confusion[ref, hyp] += 1
    num = 0
    for i in range(4):
        num += confusion[i, i]
    accuracy = num / total
    return (accuracy, confusion)

###############################################################################
# Here is the code that gets called if you type python mp2.py on command line.
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
        description='''Run MP2 using provided data directory, and show popup windows of results.''')
    parser.add_argument('-s','--steps',nargs='*',
                        help='''Perform only the specified steps (should be a space-separated list).
                        Results of any preceding steps are read from solutions.hdf5.
                        This is intended so that you can focus debugging on the step that you specify.
                        Steps are: 
                        1: todo_spectrograms,
                        2: todo_melfilters,
                        3: todo_filterbank,
                        4: todo_gtfilters,
                        5: todo_gammatone,
                        6: todo_distances,
                        7: todo_nearest_neighbor,
                        8: todo_compute_accuracy.
                        Default: do all steps''')
    parser.add_argument('--showsolutions',action='store_true',
                        help='''If you want to show solution figures, instead of 
                        computing your own figures.''')
    parser.add_argument('--nofigures',action='store_true',
                        help='''If you want to just generate results.hdf5, without making any figures.''')
    parser.add_argument('--datadir',
                        help='''Set the datadir.  Default: "data"''',
                        default='data')
    args = parser.parse_args()
    if args.steps is not None or args.showsolutions:
        solutions = h5py.File('solutions.hdf5','r')

    class PlotWindow(object):
        '''
        Pop up a window containing a matplotlib Figure.
        The NavigationToolbar2TK allows you to save the figure in a file.
        The key_press_handler  permits standard key events as described at
        https://matplotlib.org/3.3.0/users/interactive.html#key-event-handling
        '''
        def __init__(self, fig):
            if not args.nofigures:
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
    # Step 0: load the data, and show three waveforms
    train, dev, test, labels = load_datasets(args.datadir)
    nwtrain = dataset_to_matrix(train)
    nwdev = dataset_to_matrix(dev)
    nwtest = dataset_to_matrix(test)
    if (args.steps is None or '0' in args.steps) and not args.nofigures:
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(3,1,sharex=True)
        t = np.linspace(0,0.5,8000,endpoint=False)
        axs[0].plot(t,nwtrain[0,:])
        axs[0].set_title('A waveform of word "%s"'%(labels[train[0].y]))
        axs[1].plot(t,nwtrain[int(len(train)/2),:])
        axs[1].set_title('A waveform of word "%s"'%(labels[train[int(len(train)/2)].y]))
        axs[2].plot(t,nwtrain[-1,:])
        axs[2].set_title('A waveform of word "%s"'%(labels[train[-1].y]))
        axs[2].set_xlabel('Time (sec)')
        PlotWindow(fig)

    ##############################################################
    # Step 1: Compute spectrograms
    if (args.steps is None or '1' in args.steps) and not args.showsolutions:
        sgtrain, sgdev, sgtest = todo_spectrograms(nwtrain,nwdev,nwtest,16000,0.025,0.01,1024)
    else:
        sgtrain = solutions['sgtrain']
        sgdev = solutions['sgdev']
        sgtest = solutions['sgtest']

    if (args.steps is None or '2' in args.steps) and not args.nofigures:
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(3,1,sharex=True)
        fmax = 8000
        tmax = sgtrain.shape[2]/100
        X = sgtrain[0,:,:]
        im = (X-np.amin(X))/(np.amax(X)-np.amin(X))
        axs[0].imshow(im,origin='lower',aspect='auto',extent=(0,tmax,0,fmax/1000))
        axs[0].set_title('Spectrogram of word "%s"'%(labels[train[0].y]))
        axs[0].set_ylabel('Freq (kHz)')
        X = sgtrain[int(len(train)/2),:,:]
        im = (X-np.amin(X))/(np.amax(X)-np.amin(X))
        axs[1].imshow(im,origin='lower',aspect='auto',extent=(0,tmax,0,fmax/1000))
        axs[1].set_title('Spectrogram of word "%s"'%(labels[train[int(len(train)/2)].y])) 
        axs[1].set_ylabel('Freq (kHz)')
        X = sgtrain[-1,:,:]
        im = (X-np.amin(X))/(np.amax(X)-np.amin(X))
        axs[2].imshow(im,origin='lower',aspect='auto',extent=(0,tmax,0,fmax/1000))
        axs[2].set_title('Spectrogram of word "%s"'%(labels[train[-1].y]))
        axs[2].set_ylabel('Freq (kHz)')
        axs[2].set_xlabel('Time (sec)')
        PlotWindow(fig)

    ##############################################################
    # Step 2: Create mel filterbank
    if (args.steps is None or '2' in args.steps) and not args.showsolutions:
        melfilters, hertzmelcenters = todo_melfilters(sgtrain.shape[1], 40, 16000)
    else:
        melfilters = solutions['melfilters']
        hertzmelcenters = solutions['hertzmelcenters']

    if (args.steps is None or '2' in args.steps) and not args.nofigures:
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(2,1,sharex=True)
        f = np.linspace(0,8000,sgtrain.shape[1],endpoint=False)
        axs[0].plot(hertzmelcenters,[m+1 for m in range(len(hertzmelcenters))],'x')
        axs[0].set_title('Mel Filters')
        axs[0].set_ylabel('band #')
        axs[1].plot(f,np.transpose(melfilters))
        axs[1].set_ylabel('Amplitude')
        axs[1].set_xlabel('Frequency (Hz)')
        PlotWindow(fig)

    ##############################################################
    # Step 3: Compute mel filterbank coefficients
    if (args.steps is None or '3' in args.steps) and not args.showsolutions:
        fbtrain, fbdev, fbtest = todo_filterbank(sgtrain, sgdev, sgtest, melfilters)
    else:
        fbtrain = solutions['fbtrain']
        fbdev = solutions['fbdev']
        fbtest = solutions['fbtest']

    if (args.steps is None or '3' in args.steps) and not args.nofigures:
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(3,1,sharex=True)
        fmax = fbtrain.shape[1]
        tmax = fbtrain.shape[2]/100
        X = fbtrain[0,:,:]
        im = (X-np.amin(X))/(np.amax(X)-np.amin(X))
        axs[0].imshow(im,origin='lower',aspect='auto',extent=(0,tmax,0,fmax))
        axs[0].set_title('Fbank of word "%s"'%(labels[train[0].y]))
        axs[0].set_ylabel('band #')
        X = fbtrain[int(len(train)/2),:,:]
        im = (X-np.amin(X))/(np.amax(X)-np.amin(X))
        axs[1].imshow(im,origin='lower',aspect='auto',extent=(0,tmax,0,fmax))
        axs[1].set_title('Fbank of word "%s"'%(labels[train[int(len(train)/2)].y]))
        axs[1].set_ylabel('band #')
        X = fbtrain[-1,:,:]
        im = (X-np.amin(X))/(np.amax(X)-np.amin(X))
        axs[2].imshow(im,origin='lower',aspect='auto',extent=(0,tmax,0,fmax))
        axs[2].set_title('Fbank of word "%s"'%(labels[train[-1].y]))
        axs[2].set_ylabel('band #')
        axs[2].set_xlabel('Time (sec)')
        PlotWindow(fig)

    ##############################################################
    # Step 4: Create gammatone filterbank
    if (args.steps is None or '4' in args.steps) and not args.showsolutions:
        gtfilters, hertzerbcenters = todo_gtfilters(40, 16000, 0.025)
    else:
        gtfilters = solutions['gtfilters']
        hertzerbcenters = solutions['hertzerbcenters']

    if (args.steps is None or '4' in args.steps) and not args.nofigures:
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        ax = fig.subplots()
        f = np.linspace(0,8000,sgtrain.shape[1],endpoint=False)
        ax.plot(hertzmelcenters,[m+1 for m in range(len(hertzmelcenters))],'mx',
                    hertzerbcenters,[m+1 for m in range(len(hertzerbcenters))],'bx')
        ax.set_title('Mel (magenta) and ERBS (blue) band centers')
        ax.set_ylabel('band #')
        ax.set_xlabel('Frequency (Hz)')
        PlotWindow(fig)
        
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(3,1,sharex=True)
        t = np.linspace(0,gtfilters.shape[1]/16000,gtfilters.shape[1],endpoint=False)
        axs[0].plot(t,gtfilters[0,:])
        axs[0].set_title('Gammatone filter at fc=%gHz'%(hertzerbcenters[1]))
        axs[1].plot(t,gtfilters[int(len(hertzmelcenters)/2),:])
        axs[1].set_title('Gammatone filter at fc=%gHz'%(hertzerbcenters[int(1+len(hertzmelcenters)/2)]))
        axs[2].plot(t,gtfilters[-1,:])
        axs[2].set_title('Gammatone filter at fc=%gHz'%(hertzerbcenters[-2]))
        axs[2].set_xlabel('Time (sec)')
        PlotWindow(fig)

    ##############################################################
    # Step 5: Compute gammatone filterbank coefficients
    if (args.steps is None or '5' in args.steps) and not args.showsolutions:
        gttrain, gtdev, gttest = todo_gammatone(nwtrain, nwdev, nwtest, gtfilters, 16000, 0.01)
    else:
        gttrain = solutions['gttrain']
        gtdev = solutions['gtdev']
        gttest = solutions['gttest']

    if (args.steps is None or '5' in args.steps) and not args.nofigures:
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(3,1,sharex=True)
        fmax = gttrain.shape[1]
        tmax = gttrain.shape[2]/100
        X = gttrain[0,:,:]
        im = (X-np.amin(X))/(np.amax(X)-np.amin(X))
        axs[0].imshow(im,origin='lower',aspect='auto',extent=(0,tmax,0,fmax))
        axs[0].set_title('Gammatone of word "%s"'%(labels[train[0].y]))
        X = gttrain[int(len(train)/2),:,:]
        im = (X-np.amin(X))/(np.amax(X)-np.amin(X))
        axs[1].imshow(im,origin='lower',aspect='auto',extent=(0,tmax,0,fmax))
        axs[1].set_title('Gammatone of word "%s"'%(labels[train[int(len(train)/2)].y]))
        X = gttrain[-1,:,:]
        im = (X-np.amin(X))/(np.amax(X)-np.amin(X))
        axs[2].imshow(im,origin='lower',aspect='auto',extent=(0,tmax,0,fmax))
        axs[2].set_title('Gammatone of word "%s"'%(labels[train[-1].y]))
        PlotWindow(fig)
    
    ###############################################################
    # Step 6: Find and plot train-dev distances matrices for fbank and gammatone coefficients
    if (args.steps is None or '6' in args.steps) and not args.showsolutions:
        fbdist = todo_distances(fbtrain, fbdev)
        gtdist = todo_distances(fbtrain, fbdev)
    else:
        fbdist = solutions['fbdist']
        gtdist = solutions['gtdist']

    if (args.steps is None or '6' in args.steps) and not args.nofigures:
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(2,1,sharex=True)
        im = (fbdist-np.amin(fbdist))/(np.amax(fbdist)-np.amin(fbdist)+1e-6)
        axs[0].imshow(im.T)
        axs[0].set_title('fbank dist dev:train')
        axs[0].set_ylabel('dev #')
        im = (gtdist-np.amin(gtdist))/(np.amax(gtdist)-np.amin(gtdist)+1e-6)
        axs[1].imshow(im.T)
        axs[1].set_title('gammatone dist dev:train')
        axs[1].set_ylabel('dev #')
        axs[1].set_xlabel('train #')
        PlotWindow(fig)

    ###############################################################
    # Step 7: Nearest-neighbor classification
    if (args.steps is None or '7' in args.steps) and not args.showsolutions:
        fbhyps = todo_nearest_neighbor(train, fbdist)
        gthyps = todo_nearest_neighbor(train, gtdist)
    else:
        fbhyps = solutions['fbhyps']
        gthyps = solutions['gthyps']

    if (args.steps is None or '7' in args.steps) and not args.nofigures:
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(2,1,sharex=True)
        axs[0].plot(fbhyps)
        axs[0].set_title('class label (int) of each dev token: fb (above), gt (below)')
        axs[0].set_ylabel('class')
        axs[1].plot(gthyps)
        axs[1].set_ylabel('class')
        axs[1].set_xlabel('token #')
        PlotWindow(fig)

    ###############################################################
    # Step 8: Calculate confusion matrices
    if (args.steps is None or '7' in args.steps) and not args.showsolutions:
        fbacc, fbconfusion = todo_compute_accuracy(dev, fbhyps)
        gtacc, gtconfusion = todo_compute_accuracy(dev, gthyps)
    else:
        fbconfusion = solutions['fbconfusion']
        fbacc = np.sum(np.diag(fbconfusion))/np.sum(fbconfusion)
        gtconfusion = solutions['gtconfusion']
        gtacc = np.sum(np.diag(gtconfusion))/np.sum(gtconfusion)
        
    if (args.steps is None or '8' in args.steps) and not args.nofigures:
        fig = matplotlib.figure.Figure(figsize=(5, 4))
        axs = fig.subplots(2,1,sharex=True)
        axs[0].imshow(fbconfusion)
        axs[0].set_title('fbank confusion; acc=%d%%'%(int(100*fbacc)))
        axs[0].set_ylabel('ref')
        axs[1].imshow(gtconfusion)
        axs[1].set_title('gammatone confusion; acc=%d%%'%(int(100*gtacc)))
        axs[1].set_ylabel('ref')
        axs[1].set_xlabel('hyp')
        PlotWindow(fig)

    ###############################################################
    # Now create an hdf5 file with your results
    with h5py.File('results.hdf5', 'w') as f:
        f.create_dataset('nwtrain', data=nwtrain)
        f.create_dataset('nwdev', data=nwdev)
        f.create_dataset('nwtest', data=nwtest)
        f.create_dataset('sgtrain', data=sgtrain)
        f.create_dataset('sgdev', data=sgdev)
        f.create_dataset('sgtest', data=sgtest)
        f.create_dataset('fbtrain', data=fbtrain)
        f.create_dataset('fbdev', data=fbdev)
        f.create_dataset('fbtest', data=fbtest)
        f.create_dataset('gttrain', data=gttrain)
        f.create_dataset('gtdev', data=gtdev)
        f.create_dataset('gttest', data=gttest)
        f.create_dataset('melfilters', data=melfilters)
        f.create_dataset('hertzmelcenters', data=hertzmelcenters)
        f.create_dataset('gtfilters', data=gtfilters)
        f.create_dataset('hertzerbcenters', data=hertzerbcenters)
        f.create_dataset('fbdist', data=fbdist)
        f.create_dataset('gtdist', data=gtdist)
        f.create_dataset('fbhyps', data=fbhyps)
        f.create_dataset('gthyps', data=gthyps)
        f.create_dataset('fbconfusion', data=fbconfusion)
        f.create_dataset('gtconfusion', data=gtconfusion)

    print('Done!  Now try python run_tests.py.')
