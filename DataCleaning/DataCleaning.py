'''
Created on Nov 21, 2016
@ProjectTitle: Song Recommendation
@author: Jingyi Su(js5991)， Qianyu Cheng(qc510), Luyu Jin(lj1035)
'''
from package import datapreprocessing
import pandas as pd
import sys


"""
This is the main module for DataCleaning.
"""
if __name__ == '__main__':
    # Catch KeyboardInterrupt and EOFError.
    try:
        # Use the DataConverter class to create an object
        songs_data_merge = datapreprocessing.DataConverter('MillionSongSubset/data', 'lastfm_subset')
        # Save the converted DataFrame to a csv named 'song_data_merge.csv'.
        songs_data_merge.save_to_csv('song_data_merge.csv')

        # Read in a csv file named 'song_data_merge.csv'
        song_df = pd.read_csv('song_data_merge.csv', index_col=0)

        # Use the DataClean class to create an object
        cleaned_df = datapreprocessing.DataClean(song_df)
        cleaned_df.drop_missing_values()
        cleaned_df.remove_columns()
        cleaned_df.fill_missing_values()
        cleaned_df.delete_songs()
        # Save the cleaned DataFrame to a csv name 'cleanedData.csv'
        cleaned_df.save_to_csv('cleanedData.csv')
    except KeyboardInterrupt:
        sys.exit(1)
    except IOError:
        sys.exit(1)
    except EOFError:
        sys.exit(1)
