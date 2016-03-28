# encoding: utf-8
# 载入wordnet和基础模块
from nltk.corpus import wordnet as wn
from util import set1,set2,comb,write,plotpairs
from nltk.corpus import wordnet_ic
import nltk.corpus.reader.wordnet as nltk_reader
import math
import numpy as np
from sklearn import datasets, linear_model

test_all_results=[]
ic_files=['ic-bnc.dat', 'ic-brown.dat', 'ic-semcor.dat', 
          'ic-semcorraw.dat', 'ic-shaks.dat', 'ic-treebank.dat']
ics=[wordnet_ic.ic(f) for f in ic_files]
unsupport_pos=[set() for ic in ic_files]
_INF=1e300
#===========================================================
# 由排序后的样本生成rank词典
def sort2rank(pairs, key):
    pairs.sort(key=key)
    sort_pairs=pairs
    i=1
    rank_pairs={}
    for pair in sort_pairs:
        rank_pairs[pair.word1+pair.word2]=i
        i+=1
    return rank_pairs

# 按照Spearman's rank correlation coefficient计算相关系数
def rank_coeff(pairs):
    rank_human=sort2rank(pairs,key= lambda pair:pair.mean)
    rank_predict=sort2rank(pairs,key= lambda pair:pair.score)
    
    d2sum=0
    for i,pair in enumerate(pairs):
        pair.r1=rank_human[pair.word1+pair.word2]
        pair.r2=rank_predict[pair.word1+pair.word2]
        d2=(pair.r1-pair.r2)*(pair.r1-pair.r2)
        d2sum+=d2
    n=len(pairs)
    coeff = 1.0 - float(6*d2sum)/float(n*(n*n-1))
    return coeff
#===========================================================
def test_oneset(pairs,fun):
    failed_i=[]
    scores=[]
    for i,pair in enumerate(pairs):
        score=pair.sim(fun)
        if not score:
            failed_i.append(i)
        else:
            pairs[i].score=score
            scores.append(score)
            
    avrg_score=sum(scores)/len(scores)
    for i in failed_i:
        pairs[i].score=avrg_score
    
    return pairs

def test(fun,test_name):
    global test_all_results
    pairs=test_oneset(set1(),fun)
    coeff1=rank_coeff(pairs)
    
    pairs=test_oneset(set2(),fun)
    coeff2=rank_coeff(pairs)
    
    pairs=test_oneset(comb(),fun)
    coeff=rank_coeff(pairs)
    
    plotpairs(pairs,coeff,test_name)
    test_all_results.append([coeff1,coeff2,coeff])

#===========================================================
def logpath_sim(s1,s2):
    distance=s1.shortest_path_distance(s2, simulate_root=True and s1._needs_root())
    if distance is None or distance<0:
        return None
    return -math.log(distance+1)

def path(word1,word2):
    synsets1=wn.synsets(word1)
    synsets2=wn.synsets(word2)
    scores=[]
    for s1 in synsets1:
        for s2 in synsets2:
            #print s1.pos(),s2.pos()
            if s1.pos()==s2.pos():
                score=logpath_sim(s1,s2)
                if score:
                    scores.append(score)
    if not scores:
        return None
    return max(scores)

#===========================================================
#获得IC(LCS(c1,c2))
def ic1_ic2_lcs_ic(synset1,synset2,ic):
    ic1 = nltk_reader.information_content(synset1, ic)
    ic2 = nltk_reader.information_content(synset2, ic)
    #获得祖先节点
    subsumers = synset1.common_hypernyms(synset2)
    if len(subsumers) == 0:
        subsumer_ic = 0
    else:
        subsumer_ic = max(nltk_reader.information_content(s, ic) for s in subsumers)

    return ic1, ic2, subsumer_ic

def res(word1,word2):
    def res_fun(synset1,synset2,ic):
        ic1, ic2, lcs_ic = ic1_ic2_lcs_ic(synset1,synset2, ic)
        return lcs_ic
    return ic_sim(word1,word2,res_fun)

def lin(word1,word2):
    def lin_fun(synset1,synset2,ic):
        ic1, ic2, lcs_ic = ic1_ic2_lcs_ic(synset1,synset2, ic)
        return (2.0 * lcs_ic) / (ic1 + ic2)
    return ic_sim(word1,word2,lin_fun)

def jcn(word1,word2):
    def jcn_fun(synset1,synset2,ic):
        ic1, ic2, lcs_ic = ic1_ic2_lcs_ic(synset1,synset2, ic)
        if ic1 == 0 or ic2 == 0:
            return 0
        ic_difference = ic1 + ic2 - 2 * lcs_ic
        if ic_difference == 0:
            return _INF
        return 1.0 / ic_difference
    return ic_sim(word1,word2,jcn_fun)

# 三种方法的通用计算框架
def ic_sim(word1,word2,fun):
    global lc_files,ics,unsupport_pos
    synsets1=wn.synsets(word1)
    synsets2=wn.synsets(word2)
    ret_scores=[]
    for i,ic in enumerate(ics):
        scores=[]
        for s1 in synsets1:
            for s2 in synsets2:
                if s1.pos()==s2.pos() and s1._pos not in unsupport_pos[i]:
                    try:
                        icpos=ic[s1._pos]
                    except KeyError:
                        unsupport_pos[i].add(s1._pos)
                        #print  'Information content file has no entries for part-of-speech: %s in %s'%(s1._pos,ic_files[i])
                        continue
                    score=fun(s1,s2,ic)
                    if score:
                        scores.append(score)
        if not scores:
            #print 'cannot get similarity between %s and %s from %s' %(word1,word2,ic_files[i])
            continue
        ret_scores.append(max(scores))
    if not ret_scores:
        return None
    return sum(ret_scores)/len(ret_scores)

#===========================================================
# Leacock-Chodorow方法
def lch(word1,word2):
    synsets1=wn.synsets(word1)
    synsets2=wn.synsets(word2)
    scores=[]
    for s1 in synsets1:
        for s2 in synsets2:
            #print s1.pos(),s2.pos()
            if s1.pos()==s2.pos():
                score=s1.lch_similarity(s2)
                if score:
                    scores.append(score)
    if not scores:
        return None
    return max(scores)

#test(lch,'lch')

# Wu-Palmer方法
def wup(word1,word2):
    synsets1=wn.synsets(word1)
    synsets2=wn.synsets(word2)
    scores=[]
    for s1 in synsets1:
        for s2 in synsets2:
            #print s1.pos(),s2.pos()
            if s1.pos()==s2.pos():
                score=s1.wup_similarity(s2)
                if score:
                    scores.append(score)
    if not scores:
        return None
    return max(scores)
            
#===========================================================
class Model:
    def __init__(self,trainset,testset,
                feature_funs,display=True):
        self.trainset=trainset
        self.testset=testset
        self.feature_funs=feature_funs
        self.regr = linear_model.LinearRegression()
        self.display=display

    def evaluate(self, train=False):
        if train:
            pairs=self.trainset
        else:
            pairs=self.testset
        for fun in self.feature_funs:
            failed_i=[]
            scores=[]
            for i,pair in enumerate(pairs):
                score=pair.sim(fun)
                if not score:
                    failed_i.append(i)
                else:
                    pairs[i].scores.append(score)
                    scores.append(score)
            avrg_score=sum(scores)/len(scores)
            for i in failed_i:
                pairs[i].scores.append(score)
        X=np.array([pair.scores for pair in pairs])
        #print X.shape
        Y=np.array([pair.mean for pair in pairs])
        
        if train:
            self.regr.fit(X, Y)
        else:
            Y_predict=self.regr.predict(X)
            for i,pair in enumerate(pairs):
                pairs[i].score=Y_predict[i]
            coeff=rank_coeff(pairs)
            if self.display:
                plotpairs(pairs,coeff,'machine learning')
            test_all_results.append(coeff)
            return coeff
    
    def train_test(self):
        self.evaluate(True)
        coeff=self.evaluate(False)
        return coeff

def mltest_oneset(trainset,testset):
    model=Model(trainset,testset,[path,res,lin,jcn,lch,wup],False)
    model.train_test()
    return model.testset

def mltest():        
    model=Model(set1(),set2(),[path,res,lin,jcn,lch,wup])
    coeff2=model.train_test()

    model=Model(set2(),set1(),[path,res,lin,jcn,lch,wup])
    coeff1=model.train_test()

    coeff=(coeff1*153.0+coeff2*200.0)/353
    test_all_results.append([coeff1,coeff2,coeff])

#===========================================================
def test_notebook():
    test(path,'path')
    mltest()
#===========================================================
if __name__=='__main__':
    pairs=test_oneset(set1(),path)
    write(pairs,1,'set1')
    pairs=test_oneset(set2(),path)
    write(pairs,1,'set2')
    pairs=test_oneset(set1(),jcn)
    write(pairs,2,'set1')
    pairs=test_oneset(set2(),jcn)
    write(pairs,2,'set2')
    pairs=test_oneset(set1(),res)
    write(pairs,3,'set1')
    pairs=test_oneset(set2(),res)
    write(pairs,3,'set2')
    pairs=test_oneset(set1(),lin)
    write(pairs,4,'set1')
    pairs=test_oneset(set2(),lin)
    write(pairs,4,'set2')
    pairs=mltest_oneset(set2(),set1())
    write(pairs,5,'set1')
    pairs=mltest_oneset(set1(),set2())
    write(pairs,5,'set2')
