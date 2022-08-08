import pandas as pd
import numpy as np
from numpy.random import RandomState
from sklearn.feature_extraction.text import CountVectorizer
import math
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import trange
from statistics import mean
from functools import partial
import itertools
import torch
import pyro
import time
import pickle
import os 

from helper import Helper
from model import ProdLDA
from topic_coherence import TopicCoherenceScore

import hyperopt

assert pyro.__version__.startswith('1.8.1')
smoke_test = 'CI' in os.environ



def process_text(df):
    corpus = []
    docs_mat = []
    
    helper = Helper()
    chunks = helper.split_dataframe(df['body'])
    
    vectorizer = CountVectorizer(max_df=0.5, min_df=20, ngram_range=(1,3))
    print('initialised the Count Vectorizer')
    
    # vectorizer.fit(df['body'])
    # tensor = torch.from_numpy(vectorizer.transform(df['body']).toarray())
    vectorizer.fit(df['body'])
    for chunk in chunks:
        corpus.append(torch.from_numpy(vectorizer.transform(chunk).toarray()))
        time.sleep(1)
    tensor = corpus[0]
    for i in range(1, len(corpus)):
        tensor = torch.cat((tensor, corpus[i]), 0)
        time.sleep(1)
        
    vocab = pd.DataFrame(columns=['word', 'index'])
    vocab['word'] = vectorizer.get_feature_names()
    vocab['index'] = vocab.index
    
    for doc in tensor:
        docs_mat.append(np.where(doc <= 1, doc, 1))
    docs_mat = torch.from_numpy(np.array(docs_mat))
    
    return tensor, vocab, docs_mat

def compute_coherence_values(docs, vocab, docs_mat, vec_prob, num_topics, learning_rate, num_epochs, hidden):
    PERCENT = 0.2
    batch_size = 64
    prodLDA = ProdLDA(
        vocab_size = docs.shape[1],
        num_topics=num_topics,
        hidden=100,
        dropout=0.2)
    prodLDA.to(device)
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())
    num_batches = int(math.ceil(docs.shape[0] / batch_size)) if not smoke_test else 1

    bar = trange(num_epochs)
    for epoch in bar:
        running_loss = 0.0
        for i in range(num_batches):
            batch_docs = docs[i * batch_size:(i + 1) * batch_size, :]
            loss = svi.step(batch_docs)
            running_loss += loss / batch_docs.size(0)

        bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))
    
    beta = prodLDA.beta()
    top_num = 10
    coherence = TopicCoherenceScore(docs, docs_mat, top_num, vocab, beta, vec_prob)
    topic_coherence, topic_coherence_cum, top_words = coherence.driver()
    total_coherence = mean(topic_coherence_cum)
    
    
    return topic_coherence_cum, beta, top_words
    
        
# With hyperparameter optimization
# def compute_coherence_values(docs, vocab, docs_mat, num_topics, learning_rate, num_epochs, hidden):
#     PERCENT = 0.6
#     batch_size = 64
#     prodLDA = ProdLDA(
#     vocab_size=docs.shape[1],
#     num_topics=num_topics,
#     hidden=hidden if not smoke_test else 10,
#     dropout=0.2)
#     prodLDA.to(device)
    

#     optimizer = pyro.optim.Adam({"lr": learning_rate})
#     svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())
#     num_batches = int(math.ceil(docs.shape[0] / batch_size)) if not smoke_test else 1

#     bar = trange(num_epochs)
#     for epoch in bar:
#         running_loss = 0.0
#         for i in range(num_batches):
#             batch_docs = docs[i * batch_size:(i + 1) * batch_size, :]
#             loss = svi.step(batch_docs)
#             running_loss += loss / batch_docs.size(0)

#         bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))
    
#     beta = prodLDA.beta()
#     top_num = round(PERCENT*len(vocab))
    
#     coherence = TopicCoherenceScore(docs, docs_mat, top_num, vocab, beta)
#     topic_coherence, topic_coherence_cum = coherence.driver()
#     total_coherence = mean(topic_coherence_cum)
    
#     return total_coherence
    
    
    
def hyperopt_objective(docs, vocab, docs_mat, model_results, params):
    num_topics = params['topics']
    # batch_size = params['batch_size']
    learning_rate = params['lr']
    num_epochs = params['epochs']
    hidden = params['hidden']
    
    cv = compute_coherence_values(docs=docs, vocab=vocab, docs_mat=docs_mat, num_topics=num_topics, learning_rate=learning_rate, num_epochs=num_epochs, hidden=hidden)
    
    best_coherence = 1 - cv # as hyperopt minimises
    model_results['Topics'].append(num_topics)
    # model_results['batch_size'].append(batch_size)
    model_results['learning_rate'].append(learning_rate)
    model_results['num_epochs'].append(num_epochs)
    model_results['hidden'].append(hidden)
    model_results['Coherence'].append(cv)
    
    return  best_coherence


if __name__ == '__main__':
    df = pd.read_csv('/home/jupyter/Anmol/ner_news/data/test_data.csv')
    df.dropna(inplace=True)
    print('df is ready')
    
    tensor, vocab, docs_mat = process_text(df)
    
    vec_sum = torch.sum(tensor, dim=0)
    vec_prob = torch.div(vec_sum, tensor.shape[1])
    print(vec_prob)
    print(tensor.shape)
    print(docs_mat.shape)
    print(torch.nonzero(vec_prob).shape)
    
    seed = 0
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    docs = tensor.float().to(device)
    docs_mat = docs_mat.float().to(device)
    pyro.clear_param_store()
    
    print('Performing LDA and computing topic coherence score')
    topic_coherence, beta, top_words = compute_coherence_values(docs, vocab, docs_mat, vec_prob, num_topics=20, learning_rate=1e-3, num_epochs=100, hidden=100)
    print(topic_coherence)
    
    pickle.dump(beta, open('test_model.pkl', 'wb'))
    
    result_df = pd.DataFrame(columns=['top_words', 'coherence'])
    result_df['top_words'] = top_words
    result_df['coherence'] = topic_coherence
    result_df.to_csv('test_model_results.csv', index=False)
    
    print('Calculated the mean coherence')
    print('Saved the model artifacts')
    print('Execution Successful')

    
#     hyper parameter optimization
#     model_results = {
#                      'Topics': [],
#                      # 'batch_size': [],
#                      'learning_rate': [],
#                      'num_epochs': [],
#                      'hidden': [],
#                      'Coherence': [],
#                 }

#     # Hyper-parameter Finetuning
#     topics_range = [50, 100]
#     # batch_size = [32, 64, 128]
#     # learning_rate = list(np.arange(0.001, 0.01, 0.1))
#     learning_rate = [ 0.001, 0.01]
#     num_epochs = [100]
#     hidden = [200]
    
#     params_space = {
#         'topics':  hyperopt.hp.choice('topics', topics_range),
#         # 'batch_size':  hyperopt.hp.choice('batch_size', batch_size),
#         'lr': hyperopt.hp.choice('batch_size', learning_rate),
#         'epochs': hyperopt.hp.choice('num_epochs', num_epochs),
#         'hidden':  hyperopt.hp.choice('hidden', hidden),
#     } # paramter space


#     trials = hyperopt.Trials() # runs each of the trials with each of selected parameter. randomely selected one value from param_space and run all posibble combinations

#     best_param = hyperopt.fmin(
#                         partial(hyperopt_objective, docs, vocab, docs_mat, model_results),
#                         space=params_space,
#                         algo=hyperopt.tpe.suggest,
#                         max_evals=20,
#                         trials=trials,
#                         rstate=RandomState(2021)
#                         ) # minimize some score or loss
    
#     final_results = pd.DataFrame(model_results)
#     final_results.to_csv('model_results.csv', index=False)
    



