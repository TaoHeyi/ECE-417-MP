import unittest, mp5, json
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

class TestStep(unittest.TestCase):
    def setUp(self):
        self.transcript='data/transcript.txt'
        self.lexicon='data/lexicon.txt'
        self.languagemodeltexts='data/languagemodeltexts.txt'
        with open('solutions.json')  as f:
            data = json.load(f)
            self.T=data['T']
            self.Tfinal=data['Tfinal']
            self.L=data['L']
            self.Lfinal=data['Lfinal']
            self.G=data['G']
            self.Gfinal=data['Gfinal']
            self.LG=data['LG']
            self.LGfinal=data['LGfinal']
            self.TLG=data['TLG']
            self.TLGfinal=data['TLGfinal']
            self.TLG_sorted=data['TLG_sorted']
            self.TLGfinal_sorted=data['TLGfinal_sorted']
            self.bestpath=data['bestpath']
            self.alpha=data['alpha']
            self.beta=data['beta']

    @weight(6.25)
    def test_transcript2wfst(self):
        T, Tfinal =mp5.todo_transcript2wfst(self.transcript)
        T = [ [ x for x in y ] for y in T ]
        for t in self.T:
            self.assertTrue(t in T,'todo_transcript2wfst: T missing edge %s'%(t))
        for q in self.Tfinal:
            self.assertTrue(q in Tfinal,'todo_transcript2wfst: Tfinal missing state %d'%(q))

    @weight(6.25)
    def test_lexicon2wfst(self):
        L, Lfinal = mp5.todo_lexicon2wfst(self.lexicon)
        L = [[ x for x  in y ] for y  in L ]
        for t in self.L:
            self.assertTrue(t in L,'todo_lexicon2wfst: L missing edge %s'%(t))
        for q in self.Lfinal:
            self.assertTrue(q in Lfinal,'todo_lexicon2wfst: Lfinal missing state %d'%(q))

    @weight(6.25)
    def test_unigram(self):
        G, Gfinal = mp5.todo_unigram(self.languagemodeltexts,self.L)
        for t1 in self.G:
            weights = [ t2[3] for t2 in G if t2[1]==t1[1] ]
            self.assertTrue(len(weights)>0, 'todo_unigram: G missing word %s'%(t1[1]))
            self.assertTrue(np.abs(weights[0]-t1[3])<0.1,
                            'todo_unigram: %s wt %g, should be %g'%(t1[1],weights[0],t1[3]))
        for q in self.Gfinal:
            self.assertTrue(q in Gfinal,'todo_unigram: Gfinal missing state %d'%(q))

    @weight(6.25)
    def test_fstcompose(self):
        TLG, TLGfinal = mp5.todo_fstcompose(self.T,self.Tfinal,self.LG,self.LGfinal)
        n1=len(TLGfinal)
        n2=len(self.TLGfinal)
        self.assertTrue(n1==n2,'todo_fstcompose: TLGfinal has %d edges, should have %d'%(n1,n2))
        for q in self.TLGfinal:
            self.assertTrue(q in TLGfinal,'todo_fstcompose: TLGfinal missing state %d'%(q))

    @weight(6.25)
    def test_sort_topologically(self):
        TLG_sorted, TLGfinal_sorted = mp5.todo_sort_topologically(self.TLG,self.TLGfinal)
        n1=len(TLG_sorted)
        n2=len(self.TLG_sorted)
        self.assertTrue(n1==n2,'sort_topo: TLGsorted has %d edges, should have %d'%(n1,n2))
        for n,t1 in enumerate(TLG_sorted):
            self.assertTrue(t1[4]>=t1[0],'sort_topo edge %d prevstate<nextstate'%(n))
        for q in self.TLGfinal_sorted:
            self.assertTrue(q in TLGfinal_sorted,'sort_topo: TLGfinal_sorted missing state %d'%(q))

    @weight(6.25)
    def test_fstbestpath(self):
        delta, psi, bestpath = mp5.todo_fstbestpath(self.TLG_sorted,self.TLGfinal_sorted)
        n=0
        for t1,t2 in zip(bestpath,self.bestpath):
            self.assertTrue(t1[1]==t2[1],'bestpath tuple %d istr %s should be %s'%(n,t1[1],t2[1]))
            self.assertTrue(t1[2]==t2[2],'bestpath tuple %d ostr %s should be %s'%(n,t1[2],t2[2]))
            n+=1

    @weight(6.25)
    def test_fstforward(self):
        alpha = mp5.todo_fstforward(self.TLG_sorted)
        for q in self.alpha:
            e = np.abs(alpha[int(q)]-self.alpha[q])
            self.assertTrue(e<0.01,'todo_fstforward[%s]=%g, should be %g'%(q,alpha[int(q)],self.alpha[q]))

    @weight(6.25)
    def test_fstbackward(self):
        beta = mp5.todo_fstbackward(self.TLG_sorted, self.TLGfinal_sorted)
        for q in self.beta:
            if beta[int(q)]<np.inf:
                e = np.abs(beta[int(q)]-self.beta[q])
            else:
                e=0
            self.assertTrue(e<0.01,'todo_fstbackward[%s]=%g, should be %g'%(q,beta[int(q)],self.beta[q]))
