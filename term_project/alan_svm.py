#---------------------------------------------------
# File: alan_svm.py
# Authors: Bradley Mitchell, Alan Subedi, Alex Turner
# Course: CS430-01
# Date: 12/7/2023
# Python 3.8.10
#---------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedKFold, LearningCurveDisplay, learning_curve

#function that makes plot
def plot_accuracy_graph(accuracy_data, save_path=None):
    setups = [entry['setup'] for entry in accuracy_data]
    binary_accuracies = [entry['binary_accuracy'] for entry in accuracy_data]
    level_accuracies = [entry['level_accuracy'] for entry in accuracy_data]

    # Plotting Binary Classification Accuracies
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(setups, binary_accuracies, color='blue')
    plt.title('Binary Classification Accuracies')
    plt.xlabel('Input Setup')
    plt.ylabel('Accuracy (%)')

    # Plotting 5-Level Classification Accuracies
    plt.subplot(1, 2, 2)
    plt.bar(setups, level_accuracies, color='green')
    plt.title('5-Level Classification Accuracies')
    plt.xlabel('Input Setup')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

#main method that does SVM classification
def svm_model(X, y, output_text, makeVisuals):
    #split data for testing/training
    #split as 70/30 train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm_classifier = SVC(kernel='linear', C=1.0)

    #train
    svm_classifier.fit(X_train, y_train)

    #make predictions
    predictions = svm_classifier.predict(X_test)    

    #print accuracy
    accuracy = accuracy_score(y_test, predictions)
    return accuracy * 100

#helper method --> does data manipulcation for binary classification
def svm_binary(data, testCase, output_text, makeVisuals):
    X = data[testCase]
    y = data['G3']
    y = y.map(lambda g: 'Pass' if g >= 10 else 'Fail')

    return svm_model(X, y, output_text, makeVisuals)

#helper method --> does data manipulcation for 5_level classification
def svm_5_level(data, testCase, output_text, makeVisuals):
    X = data[testCase]
    y = data['G3']

    def get_letter(g: int):
        if (g >= 16): return 'A'
        elif (g >= 14): return 'B'
        elif (g >= 12): return 'C'
        elif (g >= 10): return 'D'
        else: return 'F'

    y = y.map(get_letter)

    return svm_model(X, y, output_text, makeVisuals)

#main entry point from the main method
def start_svm(preprocessed_data, FILE_PATH, COL_NAMES, INPUT_SETUP_A, INPUT_SETUP_B, INPUT_SETUP_C, INPUT_SETUP_D, makeVisuals=True):
    print("\n---------------SVM STARTS HERE (Linear Separator) ---------------\n")

    accuracy_data = []
    ###############################################################################################################
    print("BINARY CLASSIFICATION\n")

    output_text = "INPUT_SETUP_A"
    accuracy = svm_binary(preprocessed_data, INPUT_SETUP_A, output_text, makeVisuals)
    accuracy_data.append({'setup':"SETUP_A", 'binary_accuracy':accuracy, 'level_accuracy':None})
    print(f"{output_text} -->> Accuracy: {accuracy} %")

    output_text = "INPUT_SETUP_B"
    accuracy = svm_binary(preprocessed_data, INPUT_SETUP_B, output_text, makeVisuals)
    accuracy_data.append({'setup':"SETUP_B", 'binary_accuracy':accuracy, 'level_accuracy':None})
    print(f"{output_text} -->> Accuracy: {accuracy} %")

    output_text = "INPUT_SETUP_C"
    accuracy = svm_binary(preprocessed_data, INPUT_SETUP_C, output_text, makeVisuals)
    accuracy_data.append({'setup':"SETUP_C", 'binary_accuracy':accuracy, 'level_accuracy':None})
    print(f"{output_text} -->> Accuracy: {accuracy} %")

    output_text = "INPUT_SETUP_D"
    accuracy = svm_binary(preprocessed_data, INPUT_SETUP_D, output_text, makeVisuals)
    accuracy_data.append({'setup':"SETUP_D", 'binary_accuracy':accuracy, 'level_accuracy':None})
    print(f"{output_text} -->> Accuracy: {accuracy} %")
    ###############################################################################################################
    print("\n5 LEVEL CLASSIFICATION\n")

    output_text = "INPUT_SETUP_A"
    accuracy = svm_5_level(preprocessed_data, INPUT_SETUP_A, output_text, makeVisuals)
    accuracy_data[0]['level_accuracy'] = accuracy
    print(f"{output_text} -->> Accuracy: {accuracy} %")

    output_text = "INPUT_SETUP_B"
    accuracy = svm_5_level(preprocessed_data, INPUT_SETUP_B, output_text, makeVisuals)
    accuracy_data[1]['level_accuracy'] = accuracy
    print(f"{output_text} -->> Accuracy: {accuracy} %")

    output_text = "INPUT_SETUP_C"
    accuracy = svm_5_level(preprocessed_data, INPUT_SETUP_C, output_text, makeVisuals)
    accuracy_data[2]['level_accuracy'] = accuracy
    print(f"{output_text} -->> Accuracy: {accuracy} %")

    output_text = "INPUT_SETUP_D"
    accuracy = svm_5_level(preprocessed_data, INPUT_SETUP_D, output_text, makeVisuals)
    accuracy_data[3]['level_accuracy'] = accuracy
    print(f"{output_text} -->> Accuracy: {accuracy} %")
    ###############################################################################################################
    
    save_path = 'visuals/svm/svm_plot.png'
    plot_accuracy_graph(accuracy_data, save_path)

    print("\n---------------SVM ENDS HERE-------------------------------------\n")