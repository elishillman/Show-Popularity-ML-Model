#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from imdb import Cinemagoer


# In[3]:



# # Show Data
# Data was collected using [Cinemagoer](https://cinemagoer.github.io/) via imdbscrape.py and imported here.
# 
# **_Note:_** Data is from February 2023

# In[4]:


shows = pd.read_csv('../raw/shows.csv')
shows


# # MetaCritic Information

# In[5]:


meta = pd.read_csv('../raw/metacritic.csv')
meta.columns = ['name', 'metacritic', 'meta_user']
meta = meta.drop(0)


# In[6]:


ratingsdict = {'sample': [99, 8.3, 3]}
for index, row in meta.iterrows():
  split = row['name'].split(':')
  if split[0] in ratingsdict.keys():
    if (row['metacritic'] == 'tbd' or row['meta_user'] == 'tbd') == False:
      ratingsdict[split[0]][2] = ratingsdict[split[0]][2] + 1
      ratingsdict[split[0]][0] = round(((ratingsdict[split[0]][0] * (ratingsdict[split[0]][2] - 1)) + float(row['metacritic'])) / ratingsdict[split[0]][2], 2)
      ratingsdict[split[0]][1] = round(((ratingsdict[split[0]][1] * (ratingsdict[split[0]][2] - 1)) + float(row['meta_user'])) / ratingsdict[split[0]][2], 2)
  else:
    if row['metacritic'] == 'tbd' or row['meta_user'] == 'tbd':
      ratingsdict[split[0]] = [0, 0, 0]
    else:
      ratingsdict[split[0]] = [float(row['metacritic']), float(row['meta_user']), 1]


# In[7]:


name_list = []
metacritic_list = []
meta_user_list = []
for name in shows['name']:
  name_list.append(name)

for i in range(len(name_list)):
  if name_list[i] in ratingsdict.keys():
    if ratingsdict[name_list[i]][0] == 0.0:
      metacritic_list.append(np.NaN)
      meta_user_list.append(np.NaN)
    else:
      metacritic_list.append(ratingsdict[name_list[i]][0])
      meta_user_list.append(ratingsdict[name_list[i]][1])
  else:
    metacritic_list.append(np.NaN)
    meta_user_list.append(np.NaN)


# In[8]:


metacritic_array = np.array(metacritic_list)
shows['metacritic'] = metacritic_array.tolist()


# In[9]:


meta_user_array = np.array(meta_user_list)
shows['meta_user_rating'] = meta_user_array.tolist()


# In[10]:


for index, row in shows.iterrows():
  if row['meta_user_rating'] < 0.1:
    shows.at[row['meta_user_rating']] = np.NaN
  if row['metacritic'] < 0.1:
    shows.at[row['metacritic']] = np.NaN


# In[11]:


shows


# # Content Methodology Data
# (Primary SVOD)

# In[12]:


method = pd.read_csv('../raw/svod.csv')
method.columns = ['name', 'service']


# In[13]:


method


# In[14]:


shows = shows.merge(method, how = 'left', on = ['name'])
shows.drop_duplicates
shows


# # Social Media Data
# (Reddit, Twitter)

# In[15]:


reddit = pd.read_csv('../raw/reddit.csv')
reddit.columns = ['name', 'reddit_comments']


# In[16]:


shows = shows.merge(reddit, how = 'left', on = ['name'])
shows.drop_duplicates
shows


# In[17]:


twitter = pd.read_csv('../raw/twitter.csv')
twitter.columns = ['name', 'twitter_followers']


# In[18]:


shows = shows.merge(twitter, how = 'left', on = ['name'])
shows.drop_duplicates
shows


# ## Data Exploration

# ID column is dropped as it was an arbitrary number used to collect iMDB data.

# In[19]:


del shows['ids']


# ### Numeric Data

# In[20]:


numeric = shows.select_dtypes(include=[np.number])
numeric.describe().T


# * Right skewed data for Twitter and Reddit. To be expected, will potentially look to handle later.
# * Missing several values for metacritic data. If data does not appear to be significant to iMDB ranking, it will be dropped.

# In[21]:








# Fairly well distributed from 6 to 9+

# In[22]:




# Left skewed, may need to transform

# In[23]:



# Well distributed

# In[24]:








# Well distributed around 6-8

# In[25]:








# Extremely left skewed

# In[26]:








# Extremely left skewed, 0 inflated

# In[27]:









# There is no clear correlation between imdb rating and a shows popularity rank

# In[28]:









# Here we see a slight correlation, with the most popular shows having the most votes on iMDB

# In[29]:









# In[30]:









# No clear correlation in either metacritic data points, might be removed from data

# In[31]:









# We see little correlation here, a log + 1 transformation was used due to 0 inflated left skewed data

# In[32]:









# We see little correlation here, a log + 1 transformation was used due to 0 inflated left skewed data

# ### Categorical Data

# In[33]:


categorical = shows.select_dtypes(include=[object])
categorical.describe().T


# In[34]:


categorical['imdb_rank'] = shows['imdb_rank']


# In[35]:


categorical.years.value_counts()


# #### Creating a new variable 'Status' that tracks if a show is currently airing or now

# In[36]:


def series_status(years):
    if years.endswith('-'):
        return 'current'
    else:
        return 'finished'


# In[37]:


categorical.years = categorical.years.astype(str)
categorical['status'] = categorical.years.apply(series_status)


# In[38]:


categorical.status.describe()


# In[39]:








# We see that most of the shows in the iMDB top 150 are currently airing.

# In[40]:


shows['status'] = categorical['status']


# In[41]:







# Finished shows tend to be slightly higher in the popularity rankings than current shows, but the most popular shows are current

# #### Creating a new column that tracks how long a show has run for

# This is numeric data, but will be explored here.

# In[42]:


def getRuntime(years):
    start_end = years.split('-')
    try:
        if len(start_end) == 1 or start_end[1] == '':
            return 2023 - int(start_end[0]) + 1
        else:
            return int(start_end[1]) - int(start_end[0]) + 1
    except:
        return None


# In[43]:


shows.years = shows.years.astype(str)

shows['runtime'] = shows.years.apply(getRuntime)


# In[44]:


shows.runtime.describe()


# In[45]:








# In[46]:









# New shows appear to dominate the most popular shows, which makes sense. However, some long running shows are still very popular.

# In[47]:


categorical.genre.describe()


# In[48]:


categorical['genre'] = categorical['genre'].astype(str)


# In[49]:


regex = r'^\[\'([A-Z][a-z]+)'
categorical['primary_genre'] = categorical['genre'].str.extract(regex)


# Due to there being multiple genres for the majority of shows, the primary genre will be extracted from each show for similicity.

# In[50]:


categorical.primary_genre.describe()


# In[51]:








# In[52]:


shows['primary_genre'] = categorical['primary_genre']


# Action, Crime, Drama, and Comedy dominate the most popular shows. Animation also has several shows in the top 150.

# In[53]:


categorical.service.describe()


# In[54]:








# The most popular SVODs (Hulu and Netflix) dominate the most popular shows.

# ## Data Cleaning

# ### Handling NULL values

# In[55]:


nulls = shows.isnull().sum()
nulls


# Almost a third of the data has null values for Metacritic critic and user score. These columns will be deleted from the data.

# In[56]:


del shows['metacritic']
del shows['meta_user_rating']


# Two shows have null values in several columns, it is likely that these are the same shows.

# In[57]:


rows_with_null = shows[shows.isnull().any(axis=1)]
rows_with_null.head()


# The hypothesis was correct, these two shows will be removed from the data.

# In[58]:


indices = shows[shows.isin(rows_with_null.to_dict(orient='list')).all(axis=1)].index
shows = shows.drop(indices)


# In[59]:


nulls = shows.isnull().sum()
nulls


# All NULL values have been handled, and we can proceed.

# ### Handling Outliers

# In[60]:


shows.describe().T


# There are no unrealistic outliers in the data that need to be dealt with immediately
# 
# However, we see that there are some outliers in our data. Specifically with reddit_comments. This will be handled during the preprocessing step.

# ### Removing unused columns

# In[61]:


shows.head()


# name, years (as we have created two columns that use this column's data), and genre (due to primary_genre) are all features that will not be used in predicting show popularity.

# In[62]:


del shows['name']
del shows['genre']
del shows['years']


# ## Preprocessing

# We will use sklearn's built-in preprocessing capabilities to preprocess the data. We will use pipelines to do so. SimpleImputer will handle missing values. StandardScaler will be used to scale our data and hangle outliers and OneHotEncoder will one hot encode categorical data.

# In[63]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# In[64]:


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output= False))
])


# In[65]:


shows.head()


# In[66]:


categorical_cols = ['service', 'primary_genre', 'status']
numeric_cols = ['imdb_rating', 'imdb_votes', 'reddit_comments', 'twitter_followers', 'runtime']


# In[67]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor,', preprocessor)
])


# ### Applying Pipeline

# In[68]:


shows.service = shows.service.astype(str)
shows.primary_genre = shows.primary_genre.astype(str)
shows.status = shows.status.astype(str)


# In[69]:


x = shows.drop('imdb_rank', axis=1)
y = shows['imdb_rank']
preprocessed_x = pipeline.fit_transform(x)


# # Popularity Model

# Due to the small size of my dataset, I would like to use either a Linear Regression, KNN or SVM as they are likely to be more effective when working with such a dataset. I will use Grid Search Cross Validation to tine the hyperparameters and decide between the two models.
# 
# I will use classification models as opposed to regression models due to the fact that I am performing an ordinal prediction.

# In[70]:


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score


# ### Create Train Test Split

# In[71]:


x_train, x_test, y_train, y_test = train_test_split(preprocessed_x, y, test_size = 0.2, random_state=6)


# ### Define Tested Hyperparameters

# In[72]:


models = {
    'LinearRegression': LinearRegression(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC()
}


# In[73]:


params = {
    'LinearRegression': {},
    'KNN': {
        'n_neighbors': [2, 3]
    },
    'SVM': {
        'C': [0.01, 0.1, 1, 10],
        'kernel': ['linear','rbf', 'poly']
        
    }
}


# We will be using 10-fold CV to test the models.

# In[74]:


cv = KFold(n_splits=10, shuffle=True, random_state=6)


# In[75]:


grids = {}
for name, model in models.items():
    grids[name] = GridSearchCV(estimator=model, param_grid=params[name], cv=cv, scoring='neg_mean_squared_error',verbose=0)
    grids[name].fit(x_train, y_train)
    optimal_params = grids[name].best_params_
    best_score = np.sqrt(-1*grids[name].best_score_)


# The RMSE for each of the models is extremely poor. This is likely due to the lack of data and the randomness of the features. It appears to be extremely difficult to predict when a show will become popular based on a small subset of factors. However, the linear regression model preformed the best and will be used.

# ### Sample Prediction

# In[76]:


linear_model = grids['LinearRegression']


# In[79]:


def predictShow(rating, votes, service, comments, followers, status, runtime, genre):
    fake_x = x.copy()
    feature_array = [rating, votes, service, comments, followers, status, runtime, genre]
    fake_x.loc[len(fake_x) + 1] = feature_array
    preprocessed = pipeline.fit_transform(fake_x)
    prediction = linear_model.predict(preprocessed)
    normalized_prediction = 100 - (((prediction[len(prediction)-1] - 1)/(150 - 1)) * (100 - 1) + 1)
    return round(normalized_prediction,2)

# In[80]:


pred = predictShow(9.0,10000,'Hulu',100000,1000000,'current',2.0,'Comedy')
pred


# ## Conclusions

# The model preformance was extremely poor. This is overwhelmingly likely to be caused by the use of just 150 data points. In a future experiment, it would be more wise to collect a far larger amount of data when trying to model something as complex as show popularity. However, this model still has some use in predicting a TV show's popularity, but should not be taken at face value.
