import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from wordcloud import WordCloud

class Helper:
    def __init__(self):
        pass
    
    def split_dataframe(self, df, chunk_size=50):
        chunks = list()
        num_chunks = len(df) // chunk_size + 1
        for i in range(num_chunks):
            chunks.append(df[i*chunk_size:(i+1)*chunk_size])
        return chunks 
    
    def plot_word_cloud(b, ax, v, n):
        sorted_, indices = torch.sort(b, descending=True)
        df = pd.DataFrame(indices[:100].numpy(), columns=['index'])
        words = pd.merge(df, vocab[['index', 'word']],
                         how='left', on='index')['word'].values.tolist()
        sizes = (sorted_[:100] * 1000).int().numpy().tolist()
        freqs = {words[i]: sizes[i] for i in range(len(words))}
        wc = WordCloud(background_color="white", width=800, height=500)
        wc = wc.generate_from_frequencies(freqs)
        ax.set_title('Topic %d' % (n + 1))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")