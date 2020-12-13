import numpy as np
import codecs
from collections import deque

##################################################################################
def otimes(container):
    '''Usage: c = otimes((a,b))'''
    return(np.sum(container))

def oplus(container):
    '''Usage: c = oplus((a,b))'''
    m = np.amin(container)
    if m==np.inf:
        return(np.inf)
    else:
        return(m-np.log(np.sum(np.exp(m-np.array(container)))))

def todo_lexicon2wfst(filename):
    '''
    Inputs:
    filename = a string, giving the name of a file that should be read as a lexicon.
      Each line is: word phone1 phone2 ...
      You will need to split the phones, and create an transition for each.
      CAUTION: treat multiple spaces as a single space, i.e., 'a  b' -> 'a','b'
    Outputs:
    L = a list of transitions, representing a deterministic FST, with transitions in order of the lexicon.
      Each transition is a 5-tuple: (previous state, input string, output string, weight=0, next state).
      All states should be non-negative integers, beginning with state 0, created in the order
      that they are needed in order to create the lexical entries in order.
      Every entry in the lexicon should be a path from state 0, back to state 0,
      by way of N other states, where N is the number of phones in the word.
      The first N transitions in each word should have phones as input strings, and '' as output string.
      The last transition in each word should have '' as input string, and the word as output string.
      Weights should all be 0.
    Lfinal = a list of the final states of L.
    '''
    #print(filename)
    #raise NotImplementedError('You need to write this part!')

    S = []
    L = []
    Lfinal = []
    with codecs.open(filename, encoding='utf-8') as f:
        for line in f:
            ph = line.strip().split()
            S.append(ph)
    #print(S)
    #L.append((0,0,0,0))

    count = 1
    pspti ={}
    for i in S:
        cur = 0
        for j in range(1, len(i)+1):
            if j == len(i):
                L.append((cur, "", i[0], 0, 0))
                continue
            if (cur, i[j]) in pspti.keys():
                idx = pspti[(cur, i[j])]
                cur = L[idx][-1]
            else:
                L.append((cur, i[j], "", 0, count))
                pspti[(cur, i[j])] = len(L) - 1
                cur = count
                count += 1
    '''
    count = 0
    length = 0
    initial_flag = True
    found = False
    temp = 0
    next_vow = -1
    for i in S:
        for j in range(1, len(i)):
            if(initial_flag == True):
                for k in L:
                    if (k[1] == i[j]) and (j==1) and (k[0] == 0):
                        #if(i[0] == "easy"):
                        #print("Here is k:", k)
                        #print(i)
                        found = True
                        temp = k[-1]
                        next_vow = k[-1]
                        #print("I am at 64")
                        break
                    elif(k[0] == next_vow) and (j!=1) and (k[0]!=0):
                        if(k[1] == i[j]):
                            found = True
                            temp = k[-1]
                            next_vow = k[-1]
                            if(j == len(i)-1):
                                L.append((temp, "", i[0], 0, 0))
                            break
                if(found == True):
                    found = False
                    continue
            count += 1
            initial_flag = False
            if(length == 0):
                L.append((temp, i[j], "", 0, length+1))
                length = len(L)
            else:
                length = len(L)
                L.append((temp, i[j], "", 0, count))
                length += 1
            temp = L[-1][-1]
            if(j == len(i)-1):
                L.append((L[-1][-1], "", i[0], 0, 0))
        #length = 0
        temp = 0
        initial_flag = True
    
    '''



    Lfinal.append(0)
    #print(L)


    return(L, Lfinal)
                              
def todo_unigram(textfile, L):
    '''
    Inputs:
    textfile = a string, giving the name of a file that should be read as language model training text.
      CAUTION: treat multiple spaces as a single space, i.e., 'a  b' -> 'a','b'
    L = a list of WFST transitions, specifying all words in the lexicon
    Outputs:
    G = a list of transitions, specifying a unigram grammar.
      Each transition is a 5-tuple: (previous state=0, ilabel, olabel=ilabel, weight, next state=0).
      There is only one state: state 0.
      ilabel and olabel are the same; they both equal the word.
      There should be one transition for each word in the lexicon.
      The weight should be the negative log probability of that word, estimated based on its
      frequency in textfile, Laplace-smoothed with a smoothing factor of 1.
    Gfinal = a list of the final states of G.
    '''
    # raise NotImplementedError('You need to write this part!')
    G = []
    Gfinal = []
    text = {}
    total_number = 0
    with open(textfile) as f:
        for line in f:
            for word in line.split():
                if word in text.keys():
                    text[word] += 1
                else:
                    text[word] = 1
                #total_number += 1
    #print(text)

    list_L = []
    for i in L:
        if(i[-1] == 0) and (i[2] not in list_L):
            list_L.append(i[2])

    for i in list_L:
        if i in text.keys():
            total_number += (text[i] + 1)
        else:
            total_number += 1

    #print(list_L)

    for i in L:
        if(i[-1] == 0) and (i[2] in text.keys()) and (i[2] in list_L):
            prob = (text[i[2]] + 1) / total_number
            G.append((0, i[2], i[2], -np.log(prob), 0))
            #print(i[2])
            list_L.remove(i[2])
        elif(i[-1] == 0) and (i[2] in list_L):
            prob = (1) / total_number
            G.append((0, i[2], i[2], -np.log(prob), 0))

    #print(G)

    Gfinal.append(0)

    return(G, Gfinal)

def todo_transcript2wfst(filename):
    '''
    Inputs:
    filename = a string, giving the name of a file that should be read as an IPA phone transcript.
      CAUTION: treat multiple spaces as a single space, i.e., 'a  b' -> 'a','b'
    Outputs:
    T = a list of transitions.
      Each transition is a 5-tuple: (previous state, ilabel, olabel=ilabel, weight=0, next state).
      Each ilabel and olabel are the same, they equal the phone symbol.
      There should be as many transitions as there are phone symbols in the input file.
      They should proceed from state 0 to state 1, to state 2, and so on in order.
    Tfinal = a list of the final states of T.
    '''
    # raise NotImplementedError('You need to write this part!')

    S = []
    T = []
    Tfinal = []
    with codecs.open(filename, encoding='utf-8') as f:
        S = f.read().strip().split()
    #print(S)
    #print(len(S))

    count = 0
    for i in S:
        T.append((count, i, i, 0, count+1))
        count += 1
    #print(T)
    Tfinal.append(len(T))
    return(T, Tfinal)

def transitions2states(E):
    '''Find all unique state IDs referenced in a list of  transitions'''
    Q = set()
    for t in E:
        Q.add(t[0])
        Q.add(t[-1])
    return(sorted(list(Q)))

def cross_product(QA,QB):
    '''
    Usage: QCdef = cross_product(QA,QB).
    QA, QB = lists of states in A and B.
    QCdef = dict of dicts such that QCdef[qa][qb] = qc,
    where qc = qb+len(QB)*qa.
    '''
    QCdef = {}
    for qa in QA:
        QCdef[qa]={}
        for qb in QB:
            QCdef[qa][qb] = qb+len(QB)*qa
    return(QCdef)

def todo_fstcompose(A, Afinal, B, Bfinal):
    '''
    Inputs:
    A: a list of WFST transitions
    Afinal: list of the final states in A
    B: a list of WFST transitions
    Bfinal: list of the final states in B
    Output:
    C: a list of WFST transitions, composition of A and B, with state indices
       that are the cross_product of A's states and B's states.  Transitions  should be
       sorted in order of increasing previous state, subsorted in order of increasing next state.
    Cfinal: list of the final states in C, sorted
    '''
    #raise NotImplementedError('You need to write this part!')

    C = []
    Cfinal = []
    QA = transitions2states(A)
    QB = transitions2states(B)
    QC = cross_product(QA, QB)                      #QCdef = dict of dicts such that QCdef[qa][qb] = qc,
    #print(A)
    #print(B)
    #print(Afinal)
    #print(Bfinal)
    #print(transitions2statesA)
    #print(transitions2statesB)
    
    temp = {}
    for a in A:
        for b in B:
            start = QC[a[0]][b[0]]
            if (b[1] == ""):
                end = QC[a[0]][b[-1]]
                key = (start, end)
                value = (start, b[1], b[2], b[3], end)
                if key not in temp.keys():
                    temp[key] = len(C)
                    C.append(value)

            elif (a[2] == ""):
                end = QC[a[-1]][b[0]]
                key = (start, end)
                value = (start, a[1], a[2], a[3], end)
                if key not in temp.keys():
                    temp[key] = len(C)
                    C.append(value)

            elif (a[2] == b[1]):
                end = QC[a[-1]][b[-1]]
                key = (start, end)
                weight = otimes((a[3], b[3]))
                value = (start, a[1], b[2], weight, end)
                if key not in temp.keys():
                    temp[key] = len(C)
                    C.append(value)

    max_qc = len(QA)*len(QB)
    C.sort(key=lambda x: (x[0] * max_qc + x[4]))

    for i in Afinal:
        for j in Bfinal:
            Cfinal.append(QC[i][j])

    return(C, Cfinal)

def todo_sort_topologically(A, Afinal):
    '''
    Inputs:
    A: an FST that contains no cycles
    Afinal: a list of the final states in A
    Outputs:
    B: the same FST, with states renumbered so that n[t]>=p[t] for every transition.
    Bfinal: a  list of the final states in B
    '''
    #raise NotImplementedError('You need to write this part!')

    B = []
    Bfinal = []
    froniter = []
    explored = set()
    A2B = {}

    # initialization here
    froniter.append(0)
    A2B[0] = 0

    # loop through to do the bfs
    while froniter:
        pA = froniter.pop(0)
        explored.add(pA)
        for i in A:
            if i[0] == pA:
                nA = i[-1]
                pB = A2B[pA]
                if nA not in A2B.keys():
                    A2B[nA] = len(A2B)
                nB = A2B[nA]
                if nA not in froniter and nA not in explored:
                    froniter.append(nA)
                B.append((pB, i[1], i[2], i[3], nB))

    # figure out the Bfinal
    key = A2B.keys()
    for i in Afinal:
        if i not in A2B.keys():
            continue
        else:
            Bfinal.append(A2B[i])

    return(B, Bfinal)
    
def todo_fstbestpath(TLG, TLGfinal):
    '''
    Inputs:
    TLG: the composed FST, topologically sorted
    TLGfinal: the list of allowed final states
    Outputs: 
    delta: a map from states to delta surprisals
    psi: a map from states to the best preceding transition
      Tie-breaker: if two transitions into q have the same cost, psi[q] should point to
      the one tested first (the one sorted earlier by todo_sort_topologically).
    bestpath: the sequence of transitions forming the best path, in order from start to finish
    '''
    # raise NotImplementedError('You need to write this part!')

    TLG = sorted(TLG, key=lambda transition: transition[0]) # sort by previous state
    delta = {q: np.inf for q in transitions2states(TLG) }
    delta[0] = 0
    psi = {}
    bestpath = []

    for i in range(1, len(delta)):
        distance = np.inf
        for j in TLG:
            if j[-1] == i:
                cur_dist = otimes((delta[j[0]], j[3]))
                if cur_dist < distance:
                    distance = cur_dist
                    label = j
        delta[i] = distance
        psi[i] = label

    final_delta = [delta[i] for i in TLGfinal]
    final_index = np.argmin(final_delta)

    state = TLGfinal[final_index]
    while state != 0:
        bestpath.append(psi[state])
        state = psi[state][0]

    #print(bestpath)

    return(delta, psi, bestpath[::-1])

def todo_fstforward(TLG):
    '''
    Inputs:
    TLG: a list of  WFST transitions, with states sorted topologically
    Output:
    alpha: forward  surprisals of each state in TLG
    '''
    #raise NotImplementedError('You need to write this part!')

    TLG = sorted(TLG, key = lambda transition: transition[0]) # sort by previous state
    alpha = {q: np.inf for q in transitions2states(TLG)}
    alpha[0] = 0
    length = len(alpha)

    for i in range(1, length):
        for j in TLG:
            if j[-1] == i:
                times = otimes((alpha[j[0]], j[3]))
                alpha[i] = oplus((alpha[i], otimes(times)))

    return(alpha)

def todo_fstbackward(TLG, TLGfinal):
    '''
    Inputs:
    TLG: a list of  WFST transitions, with states sorted topologically
    TLGfinal: a list of the final states of TLG
    Output:
    beta: backward surprisals of each state in TLG
    '''
    #raise NotImplementedError('You need to write this part!')

    TLG = sorted(TLG, key = lambda transition: transition[-1]) # sort by next state
    beta = {q: np.inf for q in transitions2states(TLG)}
    for q in TLGfinal:
        beta[q] = 0

    length = len(beta)
    for i in range(length-1, -1, -1):
        for j in TLG:
            if j[0] == i:
                times = otimes((beta[j[-1]], j[3]))
                beta[i] = oplus((beta[i], otimes(times)))


    return(beta)

##############################################################################
def fstreestimate(TLG, L, Lfinal, alpha, beta):
    '''
    Inputs:
    TLG: a list of transitions, with state indices matching alpha and beta
    L: a list of transitions, with input and output labels matching TLG
    Lfinal: a list of the final states of L
    alpha: forward surprisal of each state in TLG
    beta: backward surprisal of each state in TLG
    Output:
    LG: a copy of L, with all of its transition weights re-estimated.
      Transition tL with input label i[tL] and output label o[tL]
      is re-estimated as the oplus, over all transitions tTLG that have the 
      same i and o labels, of alpha[p[tTLG]] otimes w[tTLG] otimes beta[n[tTLG]].
    '''
    #
    # First, find xi[istr,ostr], the expected number of translations of istr and ostr
    Sigma = set([ tlg[1] for tlg in L+TLG ])
    Omega = set([ tlg[2] for tlg in L+TLG ])
    xi = { istr:{ ostr:np.inf for ostr in Omega } for istr in Sigma }
    for tlg in TLG:
        xi[tlg[1]][tlg[2]] = oplus((xi[tlg[1]][tlg[2]], otimes((alpha[tlg[0]],tlg[3],beta[tlg[-1]]))))
    #
    # Second, accumulate the transition-probability denominators for each state in L
    gamma = { q:np.inf for q in transitions2states(L) }
    for t in L:
        gamma[t[0]] = oplus((gamma[t[0]], xi[t[1]][t[2]]))
    #
    # Third, calculate the state-conditional transition surprisals
    LG = []
    for t in L:
        if xi[t[1]][t[2]] < np.inf:
            LG.append((t[0],t[1],t[2], otimes((xi[t[1]][t[2]], -gamma[t[0]])),t[-1]))
        else:
            LG.append((t[0],t[1],t[2], np.inf, t[-1]))
    LGfinal = Lfinal.copy()
    return(LG,LGfinal)

##############################################################################
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser('Run MP5 to generate results.json.')
    parser.add_argument('-L','--lexicon',default='data/lexicon.txt',
                        help='''lexicon.  Default: "data/lexicon.txt"''')
    parser.add_argument('-G','--languagemodeltexts',default='data/languagemodeltexts.txt',
                        help='''language model texts.  Default: "data/languagemodeltexts.txt"''')
    parser.add_argument('-T','--transcript',default='data/transcript.txt',
                        help='''phonetic transcript.  Default: "data/transcript.txt"''')
    args = parser.parse_args()

    T, Tfinal =todo_transcript2wfst(args.transcript)
    L, Lfinal = todo_lexicon2wfst(args.lexicon)
    G, Gfinal = todo_unigram(args.languagemodeltexts,L)
    LG, LGfinal = todo_fstcompose(L,Lfinal,G,Gfinal)
    TLG, TLGfinal = todo_fstcompose(T,Tfinal,LG,LGfinal)
    TLG_sorted, TLGfinal_sorted = todo_sort_topologically(TLG,TLGfinal)
    delta, psi, bestpath = todo_fstbestpath(TLG_sorted,TLGfinal_sorted)
    alpha = todo_fstforward(TLG_sorted)
    beta = todo_fstbackward(TLG_sorted, TLGfinal_sorted)

    import json
    with open('results.json','w')  as f:
        data={'T':T,'Tfinal':Tfinal,'L':L,'Lfinal':Lfinal,'G':G,'Gfinal':Gfinal,
              'LG':LG,'LGfinal':LGfinal,'TLG':TLG,'TLGfinal':TLGfinal,
              'TLG_sorted':TLG_sorted,'TLGfinal_sorted':TLGfinal_sorted,
              'bestpath':bestpath,'alpha':alpha,'beta':beta}
        json.dump(data,f)
