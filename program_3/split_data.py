#---------------------------------------------------
# File: split_data.py
# Authors: Bradley Mitchell, Alan Subedi, Alex Turner, Micah Mayers
# Course: CS430-01
# Date: 10/15/2023
# Python 3.8.10
#---------------------------------------------------

import random
import pandas as pd

# randomSplit
# Function to randomly split a dataset into a training set and a validation set
# dataset: pandas DataFrame containing the dataset
# split_size: int containing the number of entrys to include in the validation set
#
# returns: tuple containing the training set DataFrame (index 0) and the validation set DataFrame (index 1)
def randomSplit(dataset: pd.DataFrame, split_size: int):
    split_training: pd.DataFrame # The portion of the input dataset used as the training dataset
    split_validation: pd.DataFrame # The portion of the input dataset used as the validation dataset
    index_list: list # List used to store the index of each dataset entry selected for the validation dataset
    split_finished = False # Flag used to track if the dataset has been sucessfully split


    # Loop to produce random splits of the dataset until a split that contains all three plant types is generated
    while split_finished == False:
        count_setosa = 0 # Number of setosa type entries in the validation set
        count_versicolor = 0 # Number of versicolor type entries in the validation set
        count_virginica = 0 # Number of virginica type entries in the validation set
        list_training = [] # List used to temporarily store dataset entries that will go in the training dataset
        list_validation = [] # List used to temporarily store dataset entries that will go in the validation dataset


        # Generate a list of unique random numbers to use as the indexes of dataset entries that will be in the validation dataset
        index_list = random.sample(range(0,150), split_size)
        index_list.sort() # Sort the entries to simplfy the logic of assigning a dataset entry to the training/validation set
        index_list_current = 0 # Index of the next entry in the list to process


        # Iterate through each entry in the dataset
        # If the index of the dataset entry matches the number stored at the current index in the index list, then add the entry to the validation set
        # Otherwise, add the entry to the training set
        for dataset_line in range(0,150):
            if (index_list_current < len(index_list)):
                if (dataset_line == index_list[index_list_current]):
                    list_validation.append(dataset.iloc[dataset_line])
                    index_list_current = index_list_current + 1 # Move to checking the next index since the current one has been found and added to the validation set
                else:
                    list_training.append(dataset.iloc[dataset_line])
            else:
                    list_training.append(dataset.iloc[dataset_line])


        # Iterate through each entry selected for the validation set to count how many entries of each plant type were selected
        for i in range(0,len(list_validation)):
            entry_type = list_validation[i].iloc[4]

            if (entry_type == "Iris-setosa"): count_setosa = count_setosa + 1
            elif (entry_type == "Iris-versicolor"): count_versicolor = count_versicolor + 1
            elif (entry_type == "Iris-virginica"): count_virginica = count_virginica + 1


        # If all three plant types were selected, then mark the split as finish and construct the dataframes that will be returned by the function
        if((count_setosa > 0) & (count_versicolor > 0) & (count_virginica > 0)):
            split_finished = True

            split_training = pd.concat(list_training, axis=1)
            split_training = split_training.T
            split_training = split_training.reset_index(drop=True)

            split_validation = pd.concat(list_validation, axis=1)
            split_validation = split_validation.T
            split_validation = split_validation.reset_index(drop=True)


    return split_training, split_validation