# encoding: utf-8
import os
import matplotlib.pyplot as plt
import numpy as np

# 定义样本对类Pair
class Pair:
    def  __init__(self,word1,word2,mean):
        self.word1=word1
        self.word2=word2
        self.mean=mean
        self.score=0.0
        self.scores=[] # for multiple scores
        self.r1=None # human rank
        self.r2=None # predict rank
        
    def __str__(self):
        return self.word1+','+self.word2+','+str(self.mean)
    
    def check_mean(self,scores):
        m=sum(scores)/float(len(scores))
        #m=float('%.2f'%m0)
        if abs(m-self.mean)>0.006:
            print 'check mean failed!'
            print self,m,m0
            return False
        return True
    
    # 用函数fun计算样本相似度分值
    def sim(self,fun):
        return fun(self.word1,self.word2)
    
def load(filename):
    fin=open(filename,'r')
    pairs=[]
    line=fin.readline()
    while True:
        line=fin.readline()
        if not line:
            break
        items=line.strip().split('\t')
        word1=items[0]
        word2=items[1]
        mean=float(items[2])
        #scores=[float(s) for s in items[3:len(items)]]
        pair=Pair(word1,word2,mean)
        #assert pair.check_mean(scores)
        pairs.append(pair)
    fin.close()
    return pairs

def set1():
    pairs=load('../wordsim353/set1.tab')
    print '%d were loaded from set1'%len(pairs)
    return pairs

def set2():
    pairs=load('../wordsim353/set2.tab')
    print '%d were loaded from set2'%len(pairs)
    return pairs

def comb():
    pairs=load('../wordsim353/combined.tab')
    print '%d were loaded from combined'%len(pairs)
    return pairs

def plotpairs(pairs,coeff,test_name):
    x=np.linspace(1,len(pairs),len(pairs))
    pairs.sort(key= lambda pair:pair.mean)
    
    human=np.zeros(len(pairs))
    predict=np.zeros(len(pairs))
    
    for i,pair in enumerate(pairs):
        human[i]=pair.r1
        predict[i]=pair.r2
    plt.figure(figsize=(8,4))
    plt.scatter(x,predict,label='$predict$',color='blue',linewidth=2)
    plt.plot(x,human,label='$human$',color='red',linewidth=2)
    plt.xlabel('word')
    plt.ylabel('rank')
    plt.title(test_name+' similarity ,  coeff = %f'%coeff)
    #plt.legend()
    plt.show()

def write(pairs,folder_i,corpus):
    fout=None
    head='Word 1,Word 2,Score (mean)\r\n'
    if corpus=='set1':
        fout=open('../results/%d/result1.csv'%folder_i,'w')
    elif corpus=='set2':
        fout=open('../results/%d/result2.csv'%folder_i,'w')
    elif corpus=='combine':
        fout=open('../combined/%d/result.csv'%folder_i,'w')
    else:
        raise Error('no such corpus.')
    
    fout.write(head)
    for pair in pairs:
        fout.write('%s,%s,%.3f\r\n'%(pair.word1,pair.word2,pair.score))
    fout.close()
    return 0



#class Embedding:
#    def __init__(self):
#        pairs=comb()
#        
#        self.emb_dict={}
#        for pair in pairs:
#            self.emb_dict[pair.word1]=[]
#            self.emb_dict[pair.word2]=[]
#            
#        embedding_path=['../glove/glove.6B.300d.txt',
#                  '../glove/glove.twitter.27B.200d.txt',
#                  '../glove/vectors.840B.300d.txt']
#        for i,name in enumerate(embedding_path):
#            for key in self.emb_dict:
#                self.emb_dict[key].append([])
#                
#            f=open(name)
#            while True:
#                line=f.readline()
#                if not line:
#                    break
#                items=line.split()
#                if self.emb_dict.has_key(items[0]):
#                    self.emb_dict[items[0]][-1]=[float(e) for e in items[1:]]
#            f.close()
#            
#    def __getitem__(self,word):
#        embs=[]
#        for word_emb_dict in self.dicts:
#            if not word_emb_dict.has_key(word):
#                embs.append([])
#            else:
#                embs.appedn(word_emb_dict[word])
#        return embs