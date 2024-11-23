import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import math
import argparse

import kagglehub

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PowerTransformer
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier

from vis import *
import json

# dat = 'https://www.change4health.gov.hk/filemanager/common/pdf/brfs/2016apr.xlsx'
# key = 'https://www.change4health.gov.hk/filemanager/common/pdf/brfs/BRFS%20April%202016_Code%20Table_e.pdf'
# with open('keys.pdf', 'wb') as pdf_file: pdf_file.write(requests.get(key).content)
# with open('data.xlsx', 'wb') as file: file.write(requests.get(dat).content)




# df = pd.read_excel('data.xlsx')
# # print(f'Total number of rows: {len(df)}') # TODO: print

# with open('map_field.json', 'r') as f:
#     map_field = json.load(f)

# df.rename(columns=map_field, inplace=True)

# # preprocessing

# df = df[~df['height'].isin([7777, 9998, 9999])]
# df = df[~df['weight'].isin([7777, 9998, 9999])]
# df = df[~df['age'].isin([998, 999])]
# # print(f'Total number of half-processed rows: {len(df)}') # TODO: print


# df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
# df['class'] = 0.        
# df.loc[(df['bmi'] >= 25.) & (df['bmi'] < 30.), 'class'] = 1
# df.loc[df['bmi'] >= 30., 'class'] = 2





dashboard = { # processed features are commented out
    # "v8": "act_vig_day",
    # "v9": "act_vig_min",
    # "v10": "act_mod_day",
    # "v11": "act_mod_min",
    # "v12": "act_walk_day",
    # "v13": "act_walk_min",
    # "v14": "act_sit_min",
    # "v15": "act_month",
    # "v16": "fruit_day",
    # "v17": "fruit_serve",
    # "v18": "veg_day",
    # "v19": "veg_serve",
    # "v20": "juice_day",
    # "v26": "smoke_status",
    # "v27": "smoke_quit",
    # "v28": "smoke_serve",
    # "v30": "drink_status",
    # "v32x": "drink_month",
    # "v33_r": "drink_type",
    # "v34x": "drink_serve",
    # "v35x": "drink_binge",
    # "v54": "age"
    }
dashboard_dem ={
    "v55": "dem_education",
    "v56": "dem_marital",
    "v57": "dem_has_job",
    "v58": "dem_job",
    "v59": "dem_labour",
    "v60": "dem_income_ind",
    "v61": "dem_income_fam",
    "v62": "dem_housing" 
}
# # handle outliers, don't know, null.
# # general screening
# for col in ['act_vig_day', 'act_mod_day', 'act_walk_day','act_month', 'fruit_day','veg_day','juice_day', 'smoke_status','drink_status']: df[col] = df[col].replace(98, 0).replace(99, 0)
# # feature-specific handling
# df['act_sit_min'] = df['act_sit_min'].replace(7777, 9997)
# for col in ['act_vig_min', 'act_mod_min', 'act_walk_min', 'act_sit_min','drink_serve']:
#     feat = df[col]
#     feat = feat[~feat.isin([9997, 9998, 9999])]
#     mode = feat.mode()[0]
#     df[col] = df[col].replace(9997, 0) # n/a means non-participation here
#     df[col] = df[col].replace(9998, mode) # replace missing values with mode
#     df[col] = df[col].replace(9999, mode)

# for col in ['fruit_serve', 'veg_serve']:
#     feat = df[col]
#     feat = feat[~feat.isin([997, 998, 999])]
#     mode = feat.mode()[0]
#     df[col] = df[col].replace(997, 0) # n/a means non-participation here
#     df[col] = df[col].replace(998, mode) # replace missing values with mode
#     df[col] = df[col].replace(999, mode)

# for col in ['smoke_quit', 'smoke_serve','drink_month','drink_binge']:
#     feat = df[col]
#     feat = feat[~feat.isin([97, 98, 99])]
#     mode = feat.mode()[0]
#     df[col] = df[col].replace(97, 0) # n/a means non-participation here
#     df[col] = df[col].replace(98, mode) # replace missing values with mode
#     df[col] = df[col].replace(99, mode)

# for col in ['fruit_day','veg_day','juice_day']: df[col] = df[col].replace(8, 0)

# df['act_month'] = df['act_month'].replace(0, 7)
# df['smoke_status'] = df['smoke_status'].replace(0, 3)
# df['smoke_quit'] = df['smoke_quit'].replace(0, 4)
# df['smoke_quit'] = df['smoke_quit'].replace(96, 0)
# df['drink_status'] = df['drink_status'].replace(2, 0)
# df['drink_month'] = df['drink_month'].replace(77, 11) # handle the degenerates, all 4 of them.
# df['drink_serve'] = df['drink_serve'].replace(777, 25)
# df['drink_binge'] = df['drink_binge'].replace(77, 10)

# df.to_csv('data_wip.csv', index=False)
# FIXME: current progress. ******************************************************************
df = pd.read_csv('data_wip.csv')

map_marital = {1: 'never', 2: 'married_child', 3: 'married_nochild', 4: 'divorced', 5: 'widowed', 6: 'null'}
map_job = {1: 'admin', 2: 'pro', 3: 'associate', 4: 'clerk', 5: 'service', 6: 'sales', 7: 'agri', 8: 'craft', 9: 'manufacture', 10: 'unskilled', 11: 'oth', 77: 'na', 99: 'null'}
map_labour = {1: 'employer', 2: 'homemaker', 3: 'unemployed', 4: 'retired', 5: 'oth', 77: 'na', 99: 'null'}
map_housing = {1: 'public_rental', 2: 'subsidised_authority', 3: 'subsidised_society', 4: 'private', 5: 'rural', 6: 'informal', 7: 'dorm', 8: 'nondomestic', 9: 'null'}

df['dem_education'] = df['dem_education'].replace(5, 4)



df['dem_marital'] = df['dem_marital'].map(map_marital)
df['dem_job'] = df['dem_job'].map(map_job)
df['dem_labour'] = df['dem_labour'].map(map_labour)
df['dem_housing'] = df['dem_housing'].map(map_housing)
# One-hot encode the specified columns
columns_to_encode = ['dem_marital', 'dem_job', 'dem_labour', 'dem_housing']
df = pd.get_dummies(df, columns=columns_to_encode)
df = df.drop(columns=['dem_marital_null', 'dem_job_null', 'dem_labour_null', 'dem_housing_null','dem_job_na','dem_labour_na'])
print(df.columns)
print(f'Total number of columns: {len(df.columns)}')

# df.to_csv('data_show.csv', index=False)

# df = df.drop(columns=['Casenumber','weight','waist',
#                       'weight_plan','smoke_plan','drink_plan','drink_future', 'drink_type',

#                       'dem_has_job','bmi','weighting'])

# print(f'Total number of valid rows: {len(df)}') # TODO: print
# print(df.head()) TODO: print



# df.to_csv('data_processed.csv', index=False)



# # drop all demographic information
# df_no_dem = df.drop(columns=['dem_education','dem_marital','dem_has_job','dem_job','dem_labour',"dem_income_ind",'dem_income_fam','dem_housing'])
# df_no_dem.to_csv('data_processed_no_dem.csv', index=False)
