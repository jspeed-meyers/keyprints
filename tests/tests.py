
import os
import sys
import unittest
import numpy as np
import pandas as pd
import hashlib
import pickle

## For checking pandas dataframe equality
from pandas.util.testing import assert_frame_equal 
from numpy.testing import assert_array_equal

## Add keyprint folder with main.py to path
sys.path.append(os.path.abspath('..'))

## Functions to test
from keyprints.parser.parser_util  import analyze
from keyprints.parser.parser_util  import create_features
from keyprints.modeling.modeling   import train
from keyprints.modeling.modeling   import predict

class CheckParser(unittest.TestCase):
	## Check methods in parser folder

	def test_analyze(self):
		"""
		Check that analyze() method produces correct results.
		NOTE: If feature set changes, then both test datasets
		must be updated. Consequently, this is an annoying test
		for iterating during prototyping. It is best to use when
		the feature set is settled.
		"""

		## Import correct results
		true_analyzed_path = "test_data/analyzed_keyprint_dataset.csv"
		true_analyzed_keyprints = pd.read_csv(true_analyzed_path)

		## Import and analyze raw keyprint dataset
		test_keyprints_path = "test_data/raw_keyprint_dataset.csv"
		test_keyprints = pd.read_csv(test_keyprints_path)
		test_analyzed_keyprints = analyze(test_keyprints)

		## Check that analyzed keyprints and correct results are equal
		check = True
		try:
			## Shape, types, and values must be true
			assert_frame_equal(true_analyzed_keyprints, test_analyzed_keyprints)
		except:
			check = False
		
		self.assertTrue(check)

	def test_create_features(self):
		"""
		Check that the create features method creates the correct results.
		This test is very similar to test_analyze but notice the code
		below uses create_features(), not analyze(). This only holds if
		the raw_keyprint_dataset.csv holds less than five seconds of
		keystrokes.

		NOTE: This test will require a new test dataset if the features
		change. This is annoying during prototyping but useful for 
		stable production.
		"""

		true_features_keyprints = np.array([.25, .5, 
											.21428571428571427, 
											.42581531362632, 
											0.1111111111111111,
											0.3233808333817773])

		## Import and create features fo raw keyprint dataset
		test_keyprints_path = "test_data/raw_keyprint_dataset.csv"
		test_keyprints = pd.read_csv(test_keyprints_path)
		test_features_keyprints = create_features(test_keyprints)

		## Lists can be directly compared for equality, which
		## is why these arrays are converted to a list
		self.assertEqual(true_features_keyprints.tolist(),
						 test_features_keyprints.tolist())


class CheckModeling(unittest.TestCase):
	## Check methods in modeling folder

	def test_train(self):
		"""
		Check that train() method produces correct results.
		"""

		## Import training dataset
		training_prints_1 = pd.read_csv("test_data/user1.csv",
			                            index_col=0)
		training_prints_2 = pd.read_csv("test_data/user2.csv",
			                            index_col=0)

		## Train model
		model = train(training_prints_1, training_prints_2, False)

		## Extract model parameters, get hash of parameters after
		## encoding and check against stored hash
		model_string = str(model.params)
		model_string_encoded = model_string.encode(encoding='UTF-8',
													errors='strict')
		model_hash = hashlib.md5(model_string_encoded).hexdigest()

		## Check computed model hash is equal to correct hash
		self.assertEqual(model_hash, "6fdb3bd7b1358f2d284d7d292e3eb02e")

	def test_predict(self):
		"""
		Check that predict() method produces correct results.
		"""

		## Import pickled model
		pickle_path = "test_data/model_pickle.pkl"
		with open(pickle_path,'rb') as f:
			model = pickle.load(f)

		## Import keyprints
		keyprint_path = "test_data/analyzed_keyprint_dataset.csv"
		keyprints = pd.read_csv(keyprint_path)

		## Predict on keyprints with model, store result]
		prediction = predict(keyprints, model)

		## Check that result is correct
		correct_prediction = ('User 2', 0.99)
		self.assertEqual(prediction, correct_prediction)