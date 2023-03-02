from create_processed_dataset import data_path
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from utils import states_abv_dict

year = 2018
race = 'house'
num_topics=25
df = pd.read_csv(data_path + str(year) + '/' + race + '_text_concat_processed.csv', error_bad_lines=False)
ads_list= df['text_concat'].fillna('').tolist()

dem_cand_names = np.char.lower(df['dem_cand'].unique().tolist()).tolist()
rep_cand_names = np.char.lower(df['rep_cand'].unique().tolist()).tolist()
removed_words = list(stopwords.words('english'))
removed_words.extend(dem_cand_names)
removed_words.extend(rep_cand_names)
removed_words.extend([state.lower() for state in list(states_abv_dict.keys())])


tokenizer = RegexpTokenizer(r'\w+')
tfidf = TfidfVectorizer(lowercase=True,
                        stop_words=removed_words,
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize)

train_data = tfidf.fit_transform(ads_list)

model=LatentDirichletAllocation(n_components=num_topics)
lda_matrix = model.fit_transform(train_data)
lda_components=model.components_

terms = tfidf.get_feature_names_out()

for index, component in enumerate(lda_components):
    zipped = zip(terms, component)
    top_terms_key=sorted(zipped, key = lambda t: t[1], reverse=True)[:14]
    top_terms_list=list(dict(top_terms_key).keys())
    print("Topic "+str(index)+": ",top_terms_list)

print(lda_matrix)
np.save('lda_probs.npy', lda_matrix)