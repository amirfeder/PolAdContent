from pathlib import Path
import glob
import pandas as pd
import numpy as np
from utils import states_abv_dict, get_last_name
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# pd.options.display.max_colwidth = 100
# pd.options.display.max_columns = 100

year = 2018
race = 'house'
data_path = str(Path.home()) + '/Data/text-iv/'
votes_dta_file = data_path + str(year) + '/' + race + '_general_distr.dta'
ads_file = data_path + 'output_base_' + str(year) + '.csv'

df_votes = pd.read_stata(votes_dta_file)
df_votes['vote_diff'] = df_votes['dem_votes_pct_major'] - df_votes['rep_votes_pct_major']
df_ads = pd.read_csv(ads_file)
states = df_votes['state'].unique()

max_features = 500
representation_len = 384
text_bow_rows = ['text_bow_'+ str(i) for i in range(max_features)]
vectorizer = CountVectorizer(max_features=max_features)
bow = vectorizer.fit_transform(df_ads['text'].values.astype('U')).toarray()
df_ads_bow = pd.DataFrame(bow, index=df_ads.index, columns=[text_bow_rows])
df_ads = pd.concat([df_ads, df_ads_bow], axis=1)
df_ads.columns = ['file', 'text']+text_bow_rows

df_combined = pd.DataFrame(columns = ['vote_diff', 'dem_cand', 'rep_cand', 'text_bow'])
for state in states:
    df_votes_state = df_votes[(df_votes['state'].str.contains(state)) & (df_votes['year'] == year)]
    for i in range(len(df_votes_state)):
        df_votes_district = df_votes_state[df_votes_state['area'] == "District " + str(i + 1)] if len(df_votes_state) > 1 else df_votes_state
        cur_dem_cand = get_last_name(df_votes_district, 'dem')
        cur_rep_cand = get_last_name(df_votes_district, 'rep')
        cur_vote_diff = df_votes_district['vote_diff'].tolist()[0] if len(df_votes_district['vote_diff'].tolist()) > 0 else 0
        cur_df_ads_dist = df_ads[df_ads['file'].str.contains('wesleyan_ads/' + str(year) + '/' + race.capitalize() + '/' + race.upper() + '_' + states_abv_dict[state] + str(i+1).zfill(2))]

        cur_ads_list = cur_df_ads_dist['text'].fillna('').values.tolist()
        cur_ads_bow = cur_df_ads_dist[text_bow_rows].sum()
        cur_text_emb = np.zeros(representation_len)
        if len(cur_ads_list) > 0:
            for ad in cur_ads_list:
                cur_text_emb += model.encode(ad)
            cur_text_emb /= len(cur_ads_list)
            cur_ads_bow /= len(cur_df_ads_dist)
        cur_text_emb = cur_text_emb.tolist()

        insert_row = {
            'vote_diff': cur_vote_diff,
            'dem_cand': cur_dem_cand,
            'rep_cand': cur_rep_cand,
        }
        for i in range(max_features): insert_row['text_bow_' + str(i)] = cur_ads_bow['text_bow_' + str(i)]
        for i in range(representation_len): insert_row['text_emb_' + str(i)] = cur_text_emb[i]
        df_combined = pd.concat([df_combined, pd.DataFrame([insert_row])])

print(df_combined)
df_combined.to_csv(data_path + str(year) + '/' + race + '_bow_emb_processed.csv')