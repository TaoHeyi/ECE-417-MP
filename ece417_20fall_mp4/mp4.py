import numpy as np
import wave

class HiddenMarkovModel(object):
    '''An object to hold a hidden Markov model'''
    def __init__(self, A, mu, var):
        self.A = A
        self.mu = mu
        self.var = var

def todo_Quniform(transcript, phn2idx, nframes):
    '''
    Input:
    transcript (list): list of phone symbols, in order
    phn2idx (nphones): a dict, mapping from phone symbols to unique sequential integer indices,
      in order of their first appearance in transcript (e.g., phn2idx['h#']=0, phn2idx['sh']=1, ...).
    nframes (scalar): length of the observation sequence, in frames.
    Output:
    Quniform = state sequence, assuming that phones are assigned in the same order as the transcript,
      and assuming that each segment is equally long on the floating-point axis.  In other words, 
      Quniform[t]==i if k'th segment is i'th phone, and (k/nsegs)<=(t/nframes)<((k+1)/nsegs).
    '''    
    Quniform = np.zeros(nframes,dtype='int')
    nsegs = len(transcript)
    #raise NotImplementedError('You need to write this part!')

    for k in range(nsegs):
        for t in range(nframes):
            if (t/nframes) >= (k/nsegs) and (t/nframes) < ((k+1)/nsegs):
                Quniform[t] = phn2idx[transcript[k]]
        #print(k/nsegs)
        #for t in nframes:

    return(Quniform)

def todo_Lambda(Quniform, X):
    '''
    Input:
    Quniform = state sequence, assuming that phones are assigned in the same order as the transcript,
      and assuming that each segment is equally long on the floating-point axis.  In other words, 
      Quniform[t]==i if k'th segment is i'th phone, and (k/nsegs)<=(t/nframes)<((k+1)/nsegs).
    X (nfeats,nframes): observations
    nstates (scalar): number of distinct states to model
    Output:
    Lambda: a hidden Markov model with the following components.
      Lambda.A[i,j] = 0 if i != j, and transcript contains no (i,j) transitions.  Otherwise,
      Lambda.A[i,j] = 1/N, where N=(1 + the number of different phones that followed the i'th phone).
      Lambda.mu[i,:] = mean of frames X[:,Quniform==i]
      Lambda.var[i,:] = variance vector of frames X[:,Quniform==i] + 1e-4 (for numerical stability)
    '''
    nfeats,nframes = X.shape
    nstates = len(set(Quniform))
    A = np.eye(nstates)
    mu = np.zeros((nstates,nfeats))
    var = np.zeros((nstates,nfeats))
    transition = {}
    #print(A)
    #raise NotImplementedError('You need to write this part!')
    #print(X)

    for i in range(nstates):
        mu[i,:] = X[:,Quniform==i].mean(1)
        var[i,:] = X[:,Quniform==i].var(1) + 10**(-4)

    for i in range(nstates):
        transition[i] = np.zeros(nstates)
    for i in range(nframes-1):
        transition[Quniform[i]][Quniform[i+1]] = 1
    for i in range(nstates):
        for j in range(nstates):
            A[i,j] = transition[i][j] / np.sum(transition[i])

    #print(X[:, Quniform==0].mean(1))
    #print(X[:, Quniform == 0].mean(-1))

    return(HiddenMarkovModel(A, mu, var))

def todo_Bscaled(X, Lambda):
    '''
    Input:
    X (nfeats,nframes): observations
    Lambda.mu (nstates, nfeats): mean vectors of each class
    Lambda.Sigma (nstates, nfeats, nfeats): covariance matrix of each class
    Output:
    logB[i,t] = log of the Gaussian pdf of X[:,t] given state i
    Bscaled[i,t] = np.exp(logB[i,t]-np.amax(logB[:,t])), i.e., scaled to a maximum of 1
    '''
    nframes = X.shape[1]
    (nstates, nfeats) = Lambda.mu.shape
    logB = np.tile(-0.5*nfeats*np.log(2*np.pi), (nstates,nframes))
    #print(logB)
    Bscaled = np.zeros((nstates,nframes))
    #print(Bscaled)
    #raise NotImplementedError('You need to write this part!')

    for i in range(nstates):
        for j in range(nframes):
            logB[i,j] = logB[i,j] - 0.5*np.log(Lambda.var[i,:] + 10**(-7)).sum() - 0.5*((X[:, j] - Lambda.mu[i,:])**2/(Lambda.var[i,:] +10**(-7))).sum()

    for i in range(nstates):
        for j in range(nframes):
            Bscaled[i,j] = np.exp(logB[i, j]-np.amax(logB[:, j]))

    return(logB, Bscaled)

def safelog(x):
    '''
    This is just log, but without the error message in case you take log(0).
    In an HMM, the zeros really need to be zeros, and really need to have a -inf log.
    This function just keeps -inf as the log of zero, without printing error messages.
    '''
    y = np.tile(-np.inf, x.shape)
    w = np.where(x!=0)
    y[w] = np.log(x[w])
    return(y)
    
def todo_logdelta(Bscaled, Lambda):
    '''
    Input:
    Bscaled (nstates, nframes): observation probabilities, scaled so max(Bhat[:,t])=1
    Lambda.A (nstates, nstates): Lambda.A[i,j] is transition probability from i to j
    Assumption:
    The first state must be state 0 (pi[0]=1, pi[i]=0 for i!=0).
    Output:
    logdelta: logdelta[i,t]=logprob of most likely sequence ending in state i at frame t
      note: in order to avoid log(0) errors, you can use the safelog function above.
    psi[i,t] = index of the state at t-1 that precedes state i at frame t
    '''
    (N,T) = Bscaled.shape
    logdelta = np.zeros((N, T))
    psi = np.zeros((N, T), dtype='int')
    #raise NotImplementedError('You need to write this part!')
    #print(np.log(Bscaled[0, 0]))
    logdelta[:, 0] = 0
    logdelta[0, 0] = 1

    for i in range(1, T):
        for j in range(N):
            #prob_ob = safelog(Bscaled[j, i])
            if Bscaled[j, i] != 0:
                prob_ob = np.log(Bscaled[j, i])
            elif Bscaled[j, i] == 0:
                prob_ob = -np.inf
            logdelta[j, i] = np.max(prob_ob + logdelta[:, i-1] + safelog(Lambda.A[:, j]))
            #logdelta[j, i] += prob_ob
            psi[j, i] = np.argmax(prob_ob + logdelta[:, i-1] + safelog(Lambda.A[:, j]))

    return(logdelta,psi)

def todo_Qstar(psi, finalstate=0):
    '''
    Input:
    psi[i,t] = index of the state at t-1 that precedes state i at frame t
    finalstate = state in which the HMM must end
    Output:
    Qstar[t] = index of the state at time t, in the maximum-posterior state sequence
    '''
    T = psi.shape[1]
    Qstar = np.zeros(T,dtype='int')
    #raise NotImplementedError('You need to write this part!')
    Qstar[T-1] = finalstate
    for t in range(T-1):
        Qstar[(T-1) - t-1] = psi[Qstar[(T-1)-t], (T-1-t)]
    return(Qstar)

def todo_alphahat(Bscaled, Lambda):
    '''
    Input:
    Bscaled (nstates, nframes): observation probability of each state at each frame
    Lambda.A (nstates, nstates): Lambda.A[i,j] is transition probability from i to j
    Assumption:
    The first state must be state 0 (pi[0]=1, pi[i]=0 for i!=0).
    Output:
    alphahat (nstates, nframes): scaled forward probability (scaled so that sum(alphahat[:,t])=1)
    G (nframes): scaling factors, so that np.cumprod(G[:(t+1)])*alphahat[:,t] = alpha[:,t]
    In order to find G[0], make the assumption that pi[0]=1, pi[i]=0 for i!=0.
    '''
    # N is the state, T is the frame
    N, T = Bscaled.shape
    alphahat = np.zeros((N,T))
    alphahat[0, 0] = 1
    G = np.zeros(T)
    G[0] = Bscaled[0, 0]
    #raise NotImplementedError('You need to write this part!')

    for i in range(1, T):
        sum_g = 0
        for j in range(N):
            sum_g += (alphahat[:, i-1] * Lambda.A[:, j] * Bscaled[j, i]).sum()
        for k in range(N):
            G[i] = sum_g
            alphahat[k, i] = (alphahat[:, i - 1] * Lambda.A[:, k] * Bscaled[k, i]).sum() / sum_g
            #if(k == 1 and i ==1):
            #    print((alphahat[:, i-1] * Lambda.A[:, k] * Bscaled[k, i]))
            #    print((alphahat[:, i - 1] * Lambda.A[:, k] * Bscaled[k, i]).sum())
            #    print(alphahat[k, i])
    return(alphahat,G)

def todo_betahat(Bscaled, Lambda):
    '''
    Input:
    Bscaled[i,t] = observation probability of state i in frame t, scaled so max(Bscaled[:,t])==1
    Lambda.A (nstates, nstates): Lambda.A[i,j] is transition probability from i to j
    Output:
    betahat[i,t]= backward probability, scaled so sum(betahat[:,t])==1
      (except in the last frame, when you can set them all to 1).
    '''
    N, T = Bscaled.shape
    betahat = np.zeros((N, T))
    betahat[:, T-1] = 1
    #raise NotImplementedError('You need to write this part!')
    for i in range(T-2, -1, -1):
        sum_g = 0
        for j in range(N):
            sum_g = sum_g + (Lambda.A[j, :] * Bscaled[:, i+1] * betahat[:, i+1]).sum()
        for k in range(N):
            betahat[k, i] = (Lambda.A[k, :] * Bscaled[:, i+1] * betahat[:, i+1]).sum()/sum_g
    return(betahat)

def todo_xi(alphahat, betahat, Bscaled, Lambda):
    '''
    Input:
    alphahat (nstates, nframes): scaled forward probability
    betahat (nstates, nframes): scaled backward probability
    Bscaled (nstates, nframes): scaled observation probability
    Lambda.A (nstates, nstates): Lambda.A[i,j] is transition probability from i to j
    Output:
    xi (nframes-1, nstates, nstates): xi[t,i,j] = p(q[t]=i,q[t+1]=j|X, Lambda)
    '''
    N,T = alphahat.shape
    xi = np.zeros((T-1,N,N))
    #raise NotImplementedError('You need to write this part!')
    for i in range(T-1):
        sum_g = 0
        for j in range(N):
            temp = (alphahat[j, i]*Lambda.A[j, :]*Bscaled[:, i+1] * betahat[:, i+1]).sum()
            sum_g = sum_g + temp
        for i1 in range(N):
            for j1 in range(N):
                xi[i, i1, j1] = Bscaled[j1, i+1] * alphahat[i1, i] * Lambda.A[i1, j1] * betahat[j1, i+1]
                xi[i, i1, j1] = xi[i, i1, j1]/sum_g

    return(xi)

def todo_Lambdaprime(xi, X):
    '''
    Input:
    xi (nframes, nstates, nstates): xi[t,i,j] = p(q[t]=i,q[t+1]=j|X, Lambda)
    X (nfeats,nframes): observations
    Intermediate variable, provided for you:
    gamma[t,i] = p(state i at time t|X,Lambda)
    Output:
    Lambdaprime is the re-estimated model:
      Lambda.A[i,j] = re-estimated transition probability from i to j
      Lambda.mu[i,:] = re-estimated mean vector for state i
      Lambda.var[i,:] = re-estimated variance vector + 1e-4 (for numerical stability)
    '''
    D, T = X.shape
    N = xi.shape[2]
    gamma = np.zeros((T,N))
    for t in range(T-1):
        gamma[t,:] = np.sum(xi[t,:,:],axis=1)
    gamma[T-1,:] = np.sum(xi[T-2,:,:],axis=0)
    Aprime = np.zeros((N,N))
    muprime = np.zeros((N,D))
    varprime = np.zeros((N,D))
    #raise NotImplementedError('You need to write this part!')

    #print(gamma)

    for i in range(N):
        divider1 = xi[:, i, :].sum()
        divider2 = gamma[:, i].sum()

        for j in range(N):
            sum_xi = xi[:, i, j].sum()
            Aprime[i, j] = sum_xi/divider1

        for k in range(D):
            muprime[i, k] = (gamma[:, i] * X[k, :]).sum()
            muprime[i, k] = muprime[i, k]/divider2
            varprime[i, k] = (gamma[:, i] * (X[k, :] - muprime[i, k])**2).sum()
            varprime[i, k] = varprime[i, k]/divider2


    return(HiddenMarkovModel(Aprime, muprime, varprime))

##############################################################################
if __name__=="__main__":
    import argparse, librosa
    parser = argparse.ArgumentParser('Run MP4 to generate results.hdf5.')
    parser.add_argument('-w','--wav_filename',default='data/LDC93S1.wav',
                        help='''wav filename.  Default: "data/LDC93S1.wav"''')
    parser.add_argument('-p','--phn_filename',default='data/LDC93S1.phn',
                        help='''phn filename.  Default: "data/LDC93S1.phn"''')
    args = parser.parse_args()

    # Load the mel spectrogram
    with wave.open(args.wav_filename,'rb') as w:
        fs = w.getframerate()
        nsamples = w.getnframes()
        wav = np.frombuffer(w.readframes(nsamples),dtype=np.int16).astype('float32')
    N = 512
    skip = int(0.01*fs)
    L = int(0.03*fs)
    magmel = librosa.feature.melspectrogram(y=wav,sr=fs,n_fft=N,hop_length=skip,
                                            power=1,win_length=L,n_mels=26) 
    X = np.log(np.maximum(1e-3,magmel/np.amax(magmel)))

    # Load the transcript, and the phn2idx dictionary
    transcript=[]
    with open(args.phn_filename) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields)>1:
                transcript.append(fields[2])
    phn2idx = {}
    for p in transcript:
        if p not in phn2idx:
            phn2idx[p]=len(phn2idx)

    Quniform = todo_Quniform(transcript, phn2idx, X.shape[1])
    Lambda = todo_Lambda(Quniform, X)
    logB, Bscaled = todo_Bscaled(X, Lambda)
    logdelta, psi = todo_logdelta(Bscaled, Lambda)
    Qstar = todo_Qstar(psi)
    alphahat, G = todo_alphahat(Bscaled, Lambda)
    betahat = todo_betahat(Bscaled, Lambda)
    xi = todo_xi(alphahat, betahat, Bscaled, Lambda)
    Lambdaprime = todo_Lambdaprime(xi, X)

    import h5py
    with h5py.File('results.hdf5','w') as f:
        f.create_dataset('X',data=X)
        f.create_dataset('Quniform',data=Quniform)
        f.create_dataset('A',data=Lambda.A)
        f.create_dataset('mu',data=Lambda.mu)
        f.create_dataset('var',data=Lambda.var)
        f.create_dataset('logB',data=logB)
        f.create_dataset('Bscaled',data=Bscaled)
        f.create_dataset('logdelta',data=logdelta)
        f.create_dataset('psi',data=psi)
        f.create_dataset('Qstar',data=Qstar)
        f.create_dataset('alphahat',data=alphahat)
        f.create_dataset('G',data=G)
        f.create_dataset('betahat',data=betahat)
        f.create_dataset('xi',data=xi)
        f.create_dataset('Aprime',data=Lambdaprime.A)
        f.create_dataset('muprime',data=Lambdaprime.mu)
        f.create_dataset('varprime',data=Lambdaprime.var)
