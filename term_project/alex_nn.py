#---------------------------------------------------
# File: alex_nn.py
# Authors: Bradley Mitchell, Alan Subedi, Alex Turner
# Course: CS430-01
# Date: 12/7/2023
# Python 3.8.10
#---------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedKFold, LearningCurveDisplay, learning_curve


# Perform both a binary and 5-level neural net classification
# All nominal variables should follow a one-hot encoding scheme
def nn_test(X: pd.DataFrame, finalGrades: pd.DataFrame, setup: str, makeVisuals: bool):

	# Binary classification
	y1 = finalGrades.map(lambda g: 'Pass' if g >= 10 else 'Fail')
	train(f'{setup}-Binary', X, y1, makeVisuals)
	
	# 5-level classification
	def get_letter(g: int):
		if (g >= 16): return 'A'
		elif (g >= 14): return 'B'
		elif (g >= 12): return 'C'
		elif (g >= 10): return 'D'
		else: return 'F'

	y2 = finalGrades.map(get_letter)
	train(f'{setup}-5-Level', X, y2, makeVisuals)


# Train and test the model with a given input scheme
def train(scheme: str, X: pd.DataFrame, y: pd.DataFrame, makeVisuals: bool):
	print(f'\nNN: {scheme} Classification')

	scaler = StandardScaler()
	clf = MLPClassifier(solver='lbfgs', max_iter=100)
	rkf = RepeatedKFold(n_splits=10, n_repeats=20)

	# visualize() uses a different method to generate a learning curve
	if makeVisuals:
		visualize(scheme, X, y, scaler, clf, rkf)
		return

	# This method reports basic metrics much faster
	trainScores, testScores = [], []

	# Perform 20 runs of a 10-fold cross-validation (200 iterations)
	for train, test in rkf.split(X):

		# 90% for training, 10% for testing
		X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]

		# Standardize all attributes
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)

		# Run the BFGS algorithm with 100 epochs
		clf.fit(X_train, y_train)
		
		# Cache metrics
		trainScores.append(clf.score(X_train, y_train))
		testScores.append(clf.score(X_test, y_test))

	# Consolidate and report metrics
	print('Training Accuracy = {:0.4f} %'.format(pd.Series(trainScores).mean() * 100))
	print('Testing Accuracy  = {:0.4f} %'.format(pd.Series(testScores).mean() * 100))


# Functionally the same as train() with the addition of learning curve plotting
# This method is considerably slower, so it should only be used when a LC is desired
def visualize(scheme: str, X: pd.DataFrame, y: pd.DataFrame, scaler: StandardScaler, clf: MLPClassifier, rkf: RepeatedKFold):

	# Generate learning curve
	# A pipeline allows standardization to occur within each fold
	pipe = Pipeline([ ('scaler', scaler), ('clf', clf) ])
	trainSizes, trainScores, testScores = learning_curve(pipe, X, y, cv=rkf)

	# Consolidate and report metrics
	print('Training Accuracy = {:0.4f} %'.format(trainScores[4].mean() * 100))
	print('Testing Accuracy  = {:0.4f} %'.format(testScores[4].mean() * 100))

	# Plot learning curve
	display = LearningCurveDisplay(train_sizes=trainSizes, train_scores=trainScores, test_scores=testScores, score_name='Accuracy')
	display.plot(line_kw={'marker': 'o'})
	plt.title(f'{scheme} Learning Curve')
	plt.legend(loc='lower right')
	plt.autoscale(False, axis='y')
	plt.ylim(0.15, 1.05)
	plt.savefig(f'./visuals/nn/{scheme}-LC.png')