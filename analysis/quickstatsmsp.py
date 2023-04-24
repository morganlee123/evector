#,Check,statistics,from,pandas,dataframe,of,labels,in,MSP
#,Author:,Morgan,Sandler

import pandas as pd

df = pd.read_csv('/research/iprobe/datastore/datasets/speech/utd-msppodcast_v1.8/Labels/labels_concensus.csv')

print(df.loc[df['Split_Set'] == 'Validation'].drop_duplicates(subset='SpkrID')['Gender'].value_counts())