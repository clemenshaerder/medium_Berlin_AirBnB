# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 19:41:26 2020

@author: chaerder
"""

#@title Load Packages
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import os, sys
import seaborn as sns
import re

print("Packages loaded")

ws = os.getcwd()

#@title Load data from listings.csv ---------
# get listings data
pd_listings = pd.read_csv(f"{ws}\\listings.csv")
list(pd_listings.columns)
# select columns from pd_listings
pd_listings = pd_listings[[
                           'id', #for joining
                           'neighbourhood_group_cleansed', # plotting
                           'latitude', # plotting
                           'longitude', # plotting
                           'number_of_reviews', # cleaning (need reviews)
                           'review_scores_rating', # mb dont need
                           'review_scores_accuracy',# mb dont need
                           'review_scores_cleanliness',# mb dont need
                           'review_scores_checkin',# mb dont need
                           'review_scores_communication',# mb dont need
                           'review_scores_location',# mb dont need
                           'review_scores_value',# mb dont need
                           # predictie features
                           'bathrooms',
                           'bedrooms',
                           'beds',
                           'room_type',
                           'price',
                           'host_has_profile_pic',
                           'host_identity_verified',
                           'host_is_superhost',
                           'host_response_rate',
                           'neighborhood_overview', # change this to length
                           'description']] # change this to length

# basic data cleaning
pd_listings['price'] = pd_listings['price'].str.replace("[$, ]", "").astype("float")
pd_listings.at[pd_listings['bathrooms'].isnull(), 'bathrooms'] = 0
pd_listings.at[pd_listings['bedrooms'].isnull(), 'bedrooms'] = 0 # there are 6 that have no bedrooms
pd_listings.at[pd_listings['beds'].isnull(), 'beds'] = 0 # there's one listing for 1 guest, without any beds
pd_listings.at[pd_listings['host_response_rate'].isnull(), 'host_response_rate'] = '0%' # there's one listing for 1 guest, without any beds
pd_listings['host_response_rate'] = pd_listings['host_response_rate'].str.replace('%', "").astype("float")

pd_listings.at[pd_listings['review_scores_rating'].isnull(), 'review_scores_rating'] = 0
pd_listings.at[pd_listings['review_scores_accuracy'].isnull(), 'review_scores_accuracy'] = 0
pd_listings.at[pd_listings['review_scores_cleanliness'].isnull(), 'review_scores_cleanliness'] = 0
pd_listings.at[pd_listings['review_scores_checkin'].isnull(), 'review_scores_checkin'] = 0
pd_listings.at[pd_listings['review_scores_communication'].isnull(), 'review_scores_communication'] = 0
pd_listings.at[pd_listings['review_scores_location'].isnull(), 'review_scores_location'] = 0
pd_listings.at[pd_listings['review_scores_value'].isnull(), 'review_scores_value'] = 0

pd_listings.rename(columns={'id':'listing_id'}, inplace=True)

# get rid of data that is not required (no reviews)
pd_listings = pd_listings[pd_listings.number_of_reviews != 0]
# also not interested in just "10s" or just "0s"
pd_listings = pd_listings[pd_listings.review_scores_rating != 100]
pd_listings = pd_listings[pd_listings.review_scores_rating != 0]

# feature engineering
pd_listings['description_len'] = pd_listings.description.str.len()
pd_listings.at[pd_listings['description_len'].isnull(), 'description_len'] = 0

pd_listings['neighborhood_overview_len'] = pd_listings.neighborhood_overview.str.len()
pd_listings.at[pd_listings['neighborhood_overview_len'].isnull(), 'neighborhood_overview_len'] = 0

print('listings.csv loaded & transformed into pd_listings')


#@title Load data from reviews.csv ---------
pd_reviews = pd.read_csv(f"{ws}\\reviews.csv")

pd_reviews = pd_reviews[['id','listing_id','date']]

# basic conversions
pd_reviews['date'] = pd.to_datetime(pd_reviews['date'])

# pd_reviews.head()
print('reviews.csv loaded into pd_reviews')

pd_listing_count_reviws = pd_reviews[['listing_id','id']].groupby(['listing_id']).count()
pd_listing_count_reviws.columns = ['# of reviews']
# pd_listing_count_reviws['listing_id'] = pd_listing_count_reviws.index

pd_listings_plus_reviews = pd.merge(pd_listings, pd_listing_count_reviws, on='listing_id')

pd_listings_plus_reviews.at[pd_listings_plus_reviews['# of reviews'].isnull(), '# of reviews'] = 0

# making sure that nothing is missing and we got all the reviews
pd_listings_plus_reviews[ pd_listings_plus_reviews['# of reviews'] != pd_listings_plus_reviews['number_of_reviews']]                              

# show rating
#@title Revenue by neighbourhood
#@title Calculate estimated revenue for each listing

# pd_bookings contains both the average & individual scores & review pre booking
# get rating by listings
pd_listings_score = pd_listings[['listing_id','review_scores_rating']].groupby(['listing_id']).sum()
# pd_listings_revenue['listing_id'] = pd_listings_revenue.index

pd_neighbourhood_score = pd_listings[['neighbourhood_group_cleansed','review_scores_value']].groupby(['neighbourhood_group_cleansed']).mean().sort_values('review_scores_value', ascending=False)

pd_listings_plot_score = pd_listings[['neighbourhood_group_cleansed','longitude','latitude','review_scores_value']]
pd_listings_plot_score.loc[:,'color'] = 0

color_value = 1
for neighbourhood in pd_neighbourhood_score[9:13].index:
  pd_listings_plot_score.at[pd_listings_plot_score['neighbourhood_group_cleansed'] == neighbourhood, 'color'] = color_value
  color_value -= 0.2

# plot
plt.figure()
ax = plt.subplot(1, 1, 1)
ax.set_title("Lowest 3 rated neighbourhoods in Berlin")

ax.set_autoscaley_on(True)

ax.set_autoscalex_on(True)

plt.scatter(pd_listings_plot_score['longitude'],
            pd_listings_plot_score['latitude'],
            cmap="coolwarm",
            c=pd_listings_plot_score['color']
           )

_ = plt.plot()

# boxplot of different categories for lowest & highest rated region
# Treptow - Köpenick                         6.716487
# Pankow                                     7.710427
list(pd_listings.columns)


scores_top_bottom = pd_listings[(pd_listings.neighbourhood_group_cleansed == 'Steglitz - Zehlendorf') | (pd_listings.neighbourhood_group_cleansed == 'Spandau')]
sns.pairplot(scores_top_bottom[['neighbourhood_group_cleansed',
                                'review_scores_accuracy',
                                'review_scores_cleanliness',
                                'review_scores_checkin',
                                'review_scores_communication',
                                'review_scores_location',
                                'review_scores_value']], hue = 'neighbourhood_group_cleansed')

corr_scores = pd_listings[['review_scores_rating',
         'review_scores_accuracy',
                           'review_scores_cleanliness',
                           'review_scores_checkin',
                           'review_scores_communication',
                           'review_scores_location',
                           'review_scores_value']].corr()

f = plt.figure(figsize=(8, 8))
corr_scores.style.background_gradient(cmap='coolwarm')
plt.matshow(corr_scores, fignum=f.number)
plt.xticks(range(corr_scores.shape[1]), corr_scores.columns, fontsize=14, rotation=45)
plt.yticks(range(corr_scores.shape[1]), corr_scores.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);

# @title Features with most weight by Linear Regression

# prep data, normalise, one-hot
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

# -----
pd_model_data_x = pd_listings[['bathrooms',
                               'bedrooms',
                               'beds',
                               'room_type',
                               'price',
                               'host_has_profile_pic',
                               'host_identity_verified',
                               'host_is_superhost',
                               'host_response_rate',
                               'neighbourhood_group_cleansed',
                               'neighborhood_overview_len',
                               'description_len']]
# simple transformations to get bathrooms, bedrooms, beds between 0 & 1
pd_model_data_x['bathrooms'] = min_max_scaler.fit_transform(pd_model_data_x[['bathrooms']])
pd_model_data_x['bedrooms'] = min_max_scaler.fit_transform(pd_model_data_x[['bedrooms']])
pd_model_data_x['beds'] = min_max_scaler.fit_transform(pd_model_data_x[['beds']])

pd_model_data_x = pd.get_dummies(pd_model_data_x, columns=['neighbourhood_group_cleansed',
                                                           'room_type',
                                                           'host_is_superhost',
                                                           'host_has_profile_pic',
                                                           'host_identity_verified'])

pd_model_data_y = pd_listings['review_scores_accuracy']
pd_model_data_y = pd_listings['review_scores_cleanliness']
pd_model_data_y = pd_listings['review_scores_checkin']
pd_model_data_y = pd_listings['review_scores_communication']
pd_model_data_y = pd_listings['review_scores_location']
pd_model_data_y = pd_listings['review_scores_value']

# train and test - x and y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pd_model_data_x,pd_model_data_y,test_size=0.10, random_state=789)

# linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

coefficients = pd.DataFrame({'feature': X_train.columns, 'Strength of linear effect': lm.coef_})
_ = coefficients.sort_values('Strength of linear effect', ascending=False)[:15].plot(x='feature', y='Strength of linear effect', kind='bar')


from sklearn.linear_model import LassoCV
X_train, X_test, y_train, y_test = train_test_split(pd_model_data_x,pd_model_data_y,test_size=0.10, random_state=789)
reg = LassoCV(cv=5, random_state=0).fit(X_train, y_train)
reg.fit(X_train, y_train)

coef_lasso = pd.DataFrame({'feature': X_train.columns, 'Strength of linear effect': reg.coef_})
_ = coef_lasso.sort_values('Strength of linear effect', ascending=False)[:15].plot(x='feature', y='Strength of linear effect', kind='bar')

forecast_lasso = reg.predict(X_test)

g=plt.scatter(y_test, forecast_lasso)
g.axes.set_xlabel('True Values ')
g.axes.set_ylabel('Predictions ')
g.axes.axis('equal')
g.axes.axis('square')



