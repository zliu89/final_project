'''
Created on Dec 4, 2016
@ProjectTitle: Song Recommendation
@author: Jingyi Su(js5991)， Qianyu Cheng(qc510), Luyu Jin(lj1035)
'''

from . import featureSelection, model, myexception
import unittest
import os.path
import pandas as pd


class Test(unittest.TestCase):
    """
    This is the class for unit tests.
    """

    def test_featureselection_init(self):
        """
        Unit test for the constructor of Feature Selection class
        """
        temp_test_data = pd.read_csv('test_data_cleaned.csv', index_col=0)
        temp_test_data_df1 = temp_test_data.ix[:, :"tags"]
        temp_test_data_df2 = temp_test_data.ix[:, "Bay Area":]
        test_data = featureSelection.FeatureSelection(temp_test_data)
        self.assertEqual(test_data.df1.shape[0], temp_test_data_df1.shape[0])
        self.assertEqual(test_data.df2.shape[0], temp_test_data_df2.shape[0])
        self.assertEqual(test_data.df1.shape[1], temp_test_data_df1.shape[1])
        self.assertEqual(test_data.df2.shape[1], temp_test_data_df2.shape[1])

    def test_remove_tag_less_than_threshold(self):
        """
        Unit test for the remove_tag_less_than_threshold function of Feature Selection class
        """
        temp_test_data = pd.read_csv('test_data_cleaned.csv', index_col=0)
        test_data = featureSelection.FeatureSelection(temp_test_data)
        test_data.remove_tag_less_than_threshold(threshold=0)
        self.assertEqual(test_data.df2.shape, (11, 40))

    def test_return_data_set(self):
        """
        Unit test for the return_data_set function of Feature Selection class
        """
        temp_test_data = pd.read_csv('test_data_cleaned.csv', index_col=0)
        test_data = featureSelection.FeatureSelection(temp_test_data)
        test_data.return_data_set()
        self.assertEqual(test_data.df1.shape[0], 11)
        self.assertEqual(test_data.df2.shape[0], 11)
        self.assertEqual(test_data.df1.shape[1] + test_data.df2.shape[1], 100)

    def test_to_csv_file(self):
        """
        Unit test for the to_csv_file function of Feature Selection class
        """
        temp_test_data = pd.read_csv('test_data_cleaned.csv', index_col=0)
        test_data = featureSelection.FeatureSelection(temp_test_data)
        test_data.to_csv_file('test_feature_selected_data.csv')
        self.assertTrue(os.path.isfile('test_feature_selected_data.csv'))

    def test_song_init(self):
        """
        Unit test for the constructor of Song class
        """
        temp_test_data = pd.read_csv('test_data_for_song.csv', index_col=0, encoding='windows-1252')
        test_data = model.Song(temp_test_data)
        self.assertEqual(test_data.textCol, ['artist_name', 'release', 'title_x', 'tags'])
        self.assertTrue(test_data.data['song_hotttnesss'].iloc[1] >= test_data.data['song_hotttnesss'].iloc[2])
        self.assertEqual(test_data.Xtrain.shape[0], 5)
        self.assertEqual(test_data.data.shape[0], 7)

    def test_lrModel(self):
        """
        Unit test for the lrModel function of Song class
        """
        temp_test_data = pd.read_csv('test_data_for_song.csv', index_col=0, encoding='windows-1252')
        test_data = model.Song(temp_test_data)
        self.assertFalse('Ypredict' in test_data.data.columns)
        test_data.Ytrain = [1, 0, 0, 1, 1]
        test_data.lrModel()
        self.assertTrue('Ypredict' in test_data.data.columns)

    def test_decisionTreeModel(self):
        """
        Unit test for the decisionTreeModel function of Song class
        """
        temp_test_data = pd.read_csv('test_data_for_song.csv', index_col=0, encoding='windows-1252')
        test_data = model.Song(temp_test_data)
        self.assertFalse('Ypredict' in test_data.data.columns)
        test_data.Ytrain = [1, 0, 0, 1, 1]
        test_data.decisionTreeModel()
        self.assertTrue('Ypredict' in test_data.data.columns)

    def test_featureSelection(self):
        """
        Unit test for the featureSelection function of Song class
        """
        temp_test_data = pd.read_csv('test_data_for_song.csv', index_col=0, encoding='windows-1252')
        test_data = model.Song(temp_test_data)
        test_data.Ytrain = [1, 0, 0, 1, 1]
        test_data.decisionTreeModel()
        self.assertTrue(test_data.model.feature_importances_ != None)

    def test_ytrainFromUser(self):
        """
        Unit test for the ytrainFromUser function of Song class
        """
        temp_test_data = pd.read_csv('test_data_for_song.csv', index_col=0, encoding='windows-1252')
        test_data = model.Song(temp_test_data)
        with self.assertRaises(myexception.InvalidInputError):
            test_data.ytrainFromUser('foo')
        with self.assertRaises(myexception.InvalidInputError):
            test_data.ytrainFromUser('1.1')





