
## Import Functions and Data


# run this cell to import nltk
import nltk
from os import getcwd
import w1_unittest
import cv2

nltk.download('twitter_samples')
nltk.download('stopwords')



filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)

import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples

from utils import process_tweet, build_freqs

### Prepare the Data


# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

"""* Train test split: 20% will be in the test set, and 80% in the training set.

"""

# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

"""* Create the numpy array of positive labels and negative labels."""

# combine positive and negative labels
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

# create frequency dictionary
freqs = build_freqs(train_x, train_y)

# UNQ_C1 GRADED FUNCTION: sigmoid
def sigmoid(z):
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''

    ### START CODE HERE ###
    # calculate the sigmoid of z

    h = 1/(1+np.exp(-z))
    ### END CODE HERE ###

    return h

# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # get 'm', the number of rows in matrix x
    m = x.shape[0]

    for i in range(0, num_iters):

        # get z, the dot product of x and theta
        z = np.dot(x,theta)

        # get the sigmoid of z
        h = sigmoid(z)

        # calculate the cost function
        J = -1./m * (np.dot(y.transpose(), np.log(h)) + np.dot((1-y).transpose(),np.log(1-h)))

        # update the weights theta
        theta = theta = theta - (alpha/m) * np.dot(x.transpose(),(h-y))

    ### END CODE HERE ###
    J = float(J)
    return J, theta

# UNQ_C3 GRADED FUNCTION: extract_features
def extract_features(tweet, freqs, process_tweet=process_tweet):
    '''
    Input:
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output:
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)

    # 3 elements for [bias, positive, negative] counts
    x = np.zeros(3)

    # bias term is set to 1
    x[0] = 1

    ### START CODE HERE ###

    # loop through each word in the list of words
    for word in word_l:

        # increment the word count for the positive label 1
        x[1] += freqs.get((word, 1.0),0)


        # increment the word count for the negative label 0
        x[2] +=  freqs.get((word, 0.0),0)


    ### END CODE HERE ###

    x = x[None, :]  # adding batch dimension for further processing
    assert(x.shape == (1, 3))
    return x

"""<a name='3'></a>
## 3 - Training Your Model

To train the model:
* Stack the features for all training examples into a matrix X.
* Call `gradientDescent`, which you've implemented above.

This section is given to you.  Please read it for understanding and run the cell.
"""

# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)



# UNQ_C4 GRADED FUNCTION: predict_tweet
def predict_tweet(tweet, freqs, theta):
   

    # extract the features of the tweet and store it into x
    x = extract_features(tweet,freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x,theta))

    ### END CODE HERE ###

    return y_pred



# UNQ_C5 GRADED FUNCTION: test_logistic_regression
def test_logistic_regression(test_x, test_y, freqs, theta, predict_tweet=predict_tweet):
   

    # the list for storing predictions
    y_hat = []

    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)

        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0.0)

   
    accuracy = (y_hat==np.squeeze(test_y)).sum()/len(test_x)

    

    return accuracy



def read_tweet_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            tweet = file.read()
            return tweet
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

# Feel free to change the tweet below
file_directory = 'C:\\Users\\admin\\Desktop\\Sentiment\\output.txt'
my_tweet = read_tweet_from_file(file_directory)
print(process_tweet(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
    image = cv2.imread('robo-removebg-preview.png')
    # Check if the image was loaded successfully
    if image is not None:
       # Display the image
       cv2.imshow('Image', image)
       cv2.waitKey(0)
       cv2.destroyAllWindows()
    else:
      print('Failed to load the image.')


else:
    print('Negative sentiment')
    image = cv2.imread('robosad-removebg-preview.png')

    # Check if the image was loaded successfully
    if image is not None:
       # Display the image
       cv2.imshow('Image', image)

       # Wait for a key press and then close the window
       cv2.waitKey(0)
       cv2.destroyAllWindows()

    else:
       print('Failed to load the image.')
