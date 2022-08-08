import pandas as pd
import numpy as np
import pyro
import pyro.distributions as dist

from sklearn.feature_extraction.text import CountVectorizer
import math
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import trange
import torch
import time
import os

assert pyro.__version__.startswith('1.8.1')
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ



class Encoder(nn.Module):
  
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout) 
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)  
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)  

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        logtheta_loc = self.bnmu(self.fcmu(h))
        logtheta_logvar = self.bnlv(self.fclv(h))
        logtheta_scale = (0.5 * logtheta_logvar).exp() 
        return logtheta_loc, logtheta_scale


class Decoder(nn.Module):
    
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)


class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.encoder = Encoder(vocab_size, num_topics, hidden, dropout)
        self.decoder = Decoder(vocab_size, num_topics, dropout)

    def model(self, docs):
        pyro.module("decoder", self.decoder)
        with pyro.plate("documents", docs.shape[0]):
            logtheta_loc = docs.new_zeros((docs.shape[0], self.num_topics))
            logtheta_scale = docs.new_ones((docs.shape[0], self.num_topics))
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            theta = F.softmax(logtheta, -1)

            # conditional distribution of 𝑤𝑛 is defined as
            # 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
            count_param = self.decoder(theta)
            total_count = int(docs.sum(-1).max())
            pyro.sample(
                'obs',
                dist.Multinomial(total_count, count_param),
                obs=docs
            )

    def guide(self, docs):
        pyro.module("encoder", self.encoder)
        with pyro.plate("documents", docs.shape[0]):
            logtheta_loc, logtheta_scale = self.encoder(docs)
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        return self.decoder.beta.weight.cpu().detach().T
    


    


# if __name__ == '__main__':
#     seed = 0
#     df = pd.read_csv('/home/jupyter/Anmol/ner_news/data/economic_times_news_data_total_processed.csv')
#     df.dropna(inplace=True)
#     print('df is ready')
    
#     docs = []
#     helper = Helper()
#     vectorizer = CountVectorizer(max_df=0.5, min_df=20, ngram_range=(1,3))
#     print('initialised the Count Vectorizer')

#     vectorizer.fit(df['body'])
#     chunks = helper.split_dataframe(df)
    
#     for chunk in chunks:
#         docs.append(torch.from_numpy(vectorizer.transform(chunk).toarray()))
#         time.sleep(2)
#     # docs = torch.from_numpy(vectorizer.fit_transform(df['body']).toarray())
#     # print(vectorised)
    
#     tensor = docs[0]
#     for i in range(1, len(docs) - 1):
#         tensor = torch.cat((tensor, docs[i]), 0)
#         time.sleep(2)
        
#     vocab = pd.DataFrame(columns=['word', 'index'])
#     vocab['word'] = vectorizer.get_feature_names()
#     vocab['index'] = vocab.index
    
#     torch.manual_seed(seed)
#     pyro.set_rng_seed(seed)
#     device = torch.device("cpu")

#     num_topics = 200 if not smoke_test else 100
#     docs = tensor.float().to(device)
    
# #     parameters
#     batch_size = 32
#     learning_rate = 1e-3
#     num_epochs = 100 if not smoke_test else 1
    
#     pyro.clear_param_store()

#     prodLDA = ProdLDA(
#         vocab_size=docs.shape[1],
#         num_topics=num_topics,
#         hidden=100 if not smoke_test else 10,
#         dropout=0.2
#     )
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
#     print(beta)
    
    
# #     if not smoke_test:
# #         import matplotlib.pyplot as plt
# #         from wordcloud import WordCloud

# #         beta = prodLDA.beta()
# #         fig, axs = plt.subplots(7, 3, figsize=(14, 24))
# #         for n in range(beta.shape[0]):
# #             i, j = divmod(n, 3)
# #             helper.plot_word_cloud(beta[n], axs[i, j], vocab, n)
# #         axs[-1, -1].axis('off');

# #         plt.show()