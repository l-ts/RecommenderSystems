#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 15:25:10 2019

@author: leo
"""

''' Import Libraries '''

import pandas as pd
import numpy as np
import scipy

''' Import Files '''

# Read data from file 'ratings.csv' located in 'data' folder (sibling of 'code' folder)
ratings = pd.read_csv("../data/ratings.csv") 
ratings.head()

''' Data Preprocessing '''

# Consider only users that have rated at least 10 movies

# calculate per user number of ratings
per_user_ratings = ratings.groupby(['userId']).size().reset_index(name='counts')
# minimum Ratings Threshold
minimumRatingsThreshold = 25
# select those with at least #(minimumRatingsThreshold) ratings
per_user_ratings = per_user_ratings[per_user_ratings['counts']>=minimumRatingsThreshold]
# filter ratings dataframe to contain only the users with userIds satisfying the minimumRatingThreshold 
ratings = ratings.merge(per_user_ratings, on='userId', how='inner')[['userId','movieId','rating']]

# Note: Further preprocessing or tuning on minimumRatingsThreshold may be required

# Note: If needed in the next questions, then the recommendations should not
# include the movies a user has already rated

''' Calculate Similarity Matrix '''

# Step 1: Write the user-item ratings data in a matrix form

''' 
For example

 	    movie1 	movie2  movie3  ...
user1 	3 	    5       1       ...
user2 	NA 	    1   	3       ...
user3 	3 	    2    	NA      ...
user4 	NA 	    NA   	2       ...
...

'''

UserItemMatrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Step 2: Create an item-to-item similarity matrix using a similarity measure


'''

Cosine Similarity Measure

To calculate similarity between items movie1 and movie2, for example,
we will consider all those users who have rated both these items:
We create two item-vectors, vector1 for item movie1 and vector2 for item movie2,
in the user-space of users (eg user1,user3,...): 
    (see matrix above where only user1 and user3 have both rated movie1 and movie2)
    vector1 = 3 * user1 + 3 * user3 + ...
    vector2 = 5 * user1 + 2 * user3 + ...
and then find the cosine of angle between these vectors. 
A zero angle would result in cosine value of 1 and means total similarity
and an angle of 90 degree would result in cosine value of 0 and means no similarity.
'''

# items are all columns of UserItemMatrix
items = UserItemMatrix.columns.tolist()


# initialise Similarity Matrix
itemSimilarityCosine = np.zeros((len(items),len(items)))

# find all 'common' userIds for every item1-item2 pair
for item1 in items:
    for item2 in items:
        # pair itemX-itemY should not be recalculated as itemY-ItemX
        if(item1<item2): # will result in an upper-triangle similarity matrix
            # find common users for these two items
            commonUsers = UserItemMatrix.dropna(subset = [ item1, item2]).index.tolist()
            # get ratings of common users for these two items 
            commonUsersRatings = UserItemMatrix.ix[commonUsers][[item1,item2]]
            # create vectors
            vector1 = commonUsersRatings[item1]
            vector2 = commonUsersRatings[item2]
            vectorsSimilarity = 1 - scipy.spatial.distance.cosine(vector1, vector2)
            itemSimilarityCosine[items.index(item1)][items.index(item2)] = vectorsSimilarity 



# Step 3: For each user, predict his rating for the items he has not rated.
'''
 
To calculate this we weigh the just-calculated similarity-measure 
between the target item and other items that the user has already rated.
The weighing factor is the ratings given by the user to items already rated by him. 
We further scale this weighted sum with the sum of similarity-measures so that the 
calculated rating remains within predefined limits.

eg 
for user3 and movie5 we would find all movies user3 has rated
and add their weighted ratings with the weights being defined
by the Similarity Matrix (where not NA) and then divide this
sum by the sum of the weights:
    
rating = (Sum(Wx,y * Rx,y) )/ (Sum(W))

 '''
 
# initialise Predictions matrix
Predictions = np.zeros((len(ratings['userId'].unique()),len(items)))

for user in ratings['userId'].unique():
    