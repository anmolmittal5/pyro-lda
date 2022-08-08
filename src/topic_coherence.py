import pandas as pd 
import numpy as np 
import torch 
from statistics import mean
import itertools

class TopicCoherenceScore:
    def __init__(self, docs, docs_mat, top_num, vocab, beta, vec_prob):
        self.docs = docs 
        self.docs_mat = docs_mat
        self.top_num = top_num
        self.vocab = vocab
        self.beta = beta
        self.vec_prob = vec_prob
        
    def segmentation(self):
        top_words = []
        top_indices = []
        vocab_arr = self.vocab.values

        for b in self.beta:
            arr_words = []
            arr_indices = []
            tmp = list(b)
            b = (sorted(b, reverse=True)[:self.top_num])
            for element in b:
                ind = tmp.index(element)
                arr_indices.append(vocab_arr[ind][1])
                arr_words.append(vocab_arr[ind][0])

            top_words.append(arr_words)
            top_indices.append(arr_indices)
            
        return top_indices, top_words
    
    def words_combination(self, top_indices):
        topics_word_pair = []
        comb_mat = []
        for top_index_list in top_indices:
            topics_word_pair.append(list(itertools.combinations(top_index_list, 2)))
        
        for i in range(0, len(topics_word_pair)):
            mat = torch.zeros((self.vocab.shape[0], len(topics_word_pair[i])))
            for j in range(0, len(topics_word_pair[i])):
                mat[topics_word_pair[i][j][0]][j] = 1
                mat[topics_word_pair[i][j][1]][j] = 1
            comb_mat.append(mat.T)  
        return comb_mat
    
    def compute_coherence(self,comb_mat, docs_mat, vec_prob):
        topic_coherence = []
        topic_coherence_cum = []
        for i in comb_mat:
            topic_score = []
            for j in i:
                topic_count = 0
                ele_p = torch.mul(vec_prob, j)
                p_w1 = float(ele_p[torch.where(j == 1)[0][0]])
                p_w2 = float(ele_p[torch.where(j == 1)[0][1]])
                for doc in docs_mat:
                    val = np.dot(doc.cpu(),j)
                    if val == 2:
                        topic_count += 1
                        
                p_w1_w2 = topic_count/ len(docs_mat)
                proba = p_w1_w2/(p_w1 * p_w2)
                topic_score.append(proba)

            topic_coherence_cum.append(mean(topic_score))
            topic_coherence.append(topic_score)
            
        return topic_coherence, topic_coherence_cum
    
    
    def driver(self):
        top_indices, top_words = self.segmentation()
        comb_mat = self.words_combination(top_indices)
        topic_coherence, topic_coherence_cum = self.compute_coherence(comb_mat, self.docs_mat, self.vec_prob)
        return topic_coherence, topic_coherence_cum, top_words
        
        
    