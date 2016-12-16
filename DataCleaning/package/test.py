'''
Created on Nov 21, 2016
@ProjectTitle: Song Recommendation
@author: Jingyi Su(js5991)， Qianyu Cheng(qc510), Luyu Jin(lj1035)
'''
import unittest
import pandas as pd
import os.path
from . import datapreprocessing

# In order to successfully pass all test, please put MillionSongSubset folder and lastfm_subsest folder in the same
# layer as the DataCleaning.py file. Otherwise, you need to manually change the path of the datafiles below.
# Also, we only use a small portion of all datafiles to run the unit test. Otherwise, it will take hours...

class Test(unittest.TestCase):
    """
    This is the class for unit tests. 
    """

    def test_dataconverter_init(self):
        """
        Unit test for the constructor of DataConverter class
        """
        test_data = datapreprocessing.DataConverter('../MillionSongSubset/data/A/A/A', '../lastfm_subset/A/A/A')
        self.assertEqual(test_data.h5_data_dir_path, '../MillionSongSubset/data/A/A/A')
        self.assertEqual(test_data.json_data_dir_path, '../lastfm_subset/A/A/A')

    def test_read_h5(self):
        """
        Unit test for the read_h5 function in DataConverter class
        """
        test_data = datapreprocessing.DataConverter('../MillionSongSubset/data/A/A/A', '../lastfm_subset/A/A/A')
        test_data_df = test_data.read_h5()
        self.assertEqual(test_data_df.shape, (33, 53))
        self.assertEqual(test_data_df.columns[0], 'analysis_sample_rate')
        self.assertEqual(test_data_df.columns[-1], 'year')

    def test_add_tags_columns(self):
        """
        Unit test for the add_tags_columns function in DataConverter class
        """
        test_data = datapreprocessing.DataConverter('../MillionSongSubset/data/A/A/A', '../lastfm_subset/A/A/A')
        test_data_df = test_data.read_json()
        test_data_df_with_tags = test_data.add_tags_columns(test_data_df)
        self.assertEqual(test_data_df_with_tags.shape, (11, 50))

    def test_read_json(self):
        """
        Unit test for the read_json function in DataConverter class
        """
        test_data = datapreprocessing.DataConverter('../MillionSongSubset/data/A/A/A', '../lastfm_subset/A/A/A')
        test_data_df = test_data.read_json()
        self.assertEqual(test_data_df.shape, (11, 50))
        self.assertEqual(test_data_df.columns[0], 'artist')
        self.assertEqual(test_data_df.columns[-1], 'Titletracks')

    def test_merge_df(self):
        """
        Unit test for the merge_df function in DataConverter class
        """
        test_data = datapreprocessing.DataConverter('../MillionSongSubset/data/A/A/A', '../lastfm_subset/A/A/A')
        test_data_df = test_data.merge_df()
        self.assertEqual(test_data_df.shape, (11, 102))
        self.assertEqual(test_data_df.columns[0], 'analysis_sample_rate')
        self.assertEqual(test_data_df.columns[-1], 'Titletracks')

    def test_dataconverter_save_to_csv(self):
        """
        Unit test for the save_to_csv function in DataConverter class
        """
        test_data = datapreprocessing.DataConverter('../MillionSongSubset/data/A/A/A', '../lastfm_subset/A/A/A')
        test_data.save_to_csv('test_data.csv')
        self.assertTrue(os.path.isfile('test_data.csv'))
        temp_df = pd.read_csv('test_data.csv', index_col=0)
        self.assertEqual(temp_df.shape, (11, 102))

    def test_dataclean_init(self):
        """
        Unit test for the constructor of DataClean class
        """
        temp_test_data = pd.read_csv('test_data.csv', index_col=0)
        test_data = datapreprocessing.DataClean(temp_test_data)
        self.assertEqual(test_data.data.shape, temp_test_data.shape)

    def test_drop_missing_values(self):
        """
        Unit test for the drop_missing_values function in DataClean class
        """
        temp_test_data = pd.read_csv('test_data.csv', index_col=0)
        test_data = datapreprocessing.DataClean(temp_test_data)
        self.assertTrue('analyzer_version' in test_data.data.columns)
        test_data.drop_missing_values()
        self.assertFalse('analyzer_version' in test_data.data.columns)

    def test_remove_columns(self):
        """
        Unit test for the remove_columns function in DataClean class
        """
        temp_test_data = pd.read_csv('test_data.csv', index_col=0)
        test_data = datapreprocessing.DataClean(temp_test_data)
        to_remove = ['analysis_sample_rate', 'audio_md5', 'track_id', 'track_7digitalid', 'artist_7digitalid',
                     'artist_id', 'release_7digitalid', 'artist_playmeid', 'artist_mbid', 'song_id', 'artist_longitude',
                     'artist_location', 'artist_latitude', 'title_y', 'artist', 'similars', 'timestamp']
        for col in set(to_remove) & set(test_data.data.columns):
            self.assertTrue(col, test_data.data.columns)
        test_data.remove_columns()
        for col in set(to_remove) & set(test_data.data.columns):
            self.assertFalse(col, test_data.data.columns)

    def test_fill_missing_values(self):
        """
        Unit test for the fill_missing_values function in DataClean class
        """
        temp_test_data = pd.read_csv('test_data.csv', index_col=0)
        test_data = datapreprocessing.DataClean(temp_test_data)
        test_data.fill_missing_values()
        self.assertFalse(test_data.data.isnull().any().any())

    def test_delete_songs(self):
        """
        Unit test for the delete_songs function in DataClean class
        """
        temp_test_data = pd.read_csv('test_data.csv', index_col=0)
        test_data = datapreprocessing.DataClean(temp_test_data)
        self.assertEqual(test_data.data.shape, (11, 102))
        test_data.delete_songs()
        self.assertEqual(test_data.data.shape, (6, 102))

    def test_dataclean_save_to_csv(self):
        """
        Unit test for the save_to_csv function in DataClean class
        """
        temp_test_data = pd.read_csv('test_data.csv', index_col=0)
        test_data = datapreprocessing.DataClean(temp_test_data)
        test_data.save_to_csv('test_data_cleaned.csv')
        self.assertTrue(os.path.isfile('test_data_cleaned.csv'))


