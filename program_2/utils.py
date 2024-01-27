
import numpy as np
import pandas as pd

def load_data(file_path, column_names):
    data = []

    # Remove header and concatenate hanging rows
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for index in range(22, len(lines), 2):
            data.append(lines[index] + lines[index + 1])

    #convert the np data to a panda dataframe and return that data
    return pd.DataFrame(np.genfromtxt(data), columns=column_names)

