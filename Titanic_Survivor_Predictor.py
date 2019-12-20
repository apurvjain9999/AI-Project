#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Naive Bayes On The Titanic Dataset
from csv import reader
from random import seed
from random import randrange
from math import sqrt
from math import exp
from math import pi
import pandas as pd
import numpy as np
from math import ceil

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# In[11]:


filename = 'Titanic.csv'
ds = load_csv(filename)
#ds
dataset = list()
i = 0
# Eliminating the 1st Column (Name) and the Row containing Column Names
for row in ds:
    if i == 0:
        i = i + 1
        continue
    dataset.append(row[1:])
#dataset


# In[12]:


# Data Cleaning
def data_clean(dataset):
    s = 0
    count = 0

    for row in dataset:
        # Converting age from float to int  
        if row[1] != "":
            row[1] = float(row[1])
            row[1] = ceil(row[1])
            s = s + row[1]
            count = count + 1
        
        # Converting Passenger Class from string to Integer value
        if row[0] == "1st":
            row[0] = int(1)
        elif row[0] == "2nd":
            row[0] = int(2)
        else:
            row[0] = int(3)
        
        # Converting gender from string to Integer value
        if row[2] == "male":
            row[2] = int(0)
        else: 
            row[2] = int(1)
        
        # Converting 
        row[3] = int(row[3])
    
    # Calulating mean age of non-missing values
    m = ceil(s / count) 

    #print(m)
    
    # Replacing the missing age values with the mean age    
    for row in dataset:
        if row[1] == "":
            row[1] = int(m)


# In[13]:


data_clean(dataset)
#dataset


# In[14]:


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers)/float(len(numbers))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)

# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries

# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities

# Predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

# Naive Bayes Algorithm
def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return(predictions)


# In[33]:


n_folds = 6
scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


# In[32]:


# fit model
model = summarize_by_class(dataset)
# define a new record
row = list()
print("Titanic Survivor Predictor")
pcl = input("\nEnter Passenger class: ")
age = input("Enter the age of the passenger: ")
gender = input("Enter the gender of the passenger: ")

if pcl == "1st" or pcl == "1":
    row.append(1)
elif pcl == "2nd" or pcl == "2":
    row.append(2)
elif pcl == "3rd" or pcl == "3":
    row.append(3)

row.append(int(age))

if gender == "M" or gender == "m" or gender == "male" or gender == "Male":
    row.append(0)
elif gender == "F" or gender == "f" or gender == "female" or gender == "Female":
    row.append(1)

# predict the label
label = predict(model, row)
sur = ""
if label == 0:
    sur = "Sorry, you are not alive !!!"
else:
    sur = "You are a SURVIVOR !!!"
print('\n' + sur)


# In[ ]:





# In[ ]:




