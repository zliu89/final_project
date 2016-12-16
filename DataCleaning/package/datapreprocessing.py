"""

"""
import pandas as pd
import json
import codecs
import os
from pandas.io.json import json_normalize


class DataConverter(object):
    """
    This class converts all datafiles in h5 format and json format into a merged pandas DataFrame.
    """

    def __init__(self, h5_data_dir_path, json_data_dir_path):
        """
        This is the constructor of the DataConverter class.
        """
        self.h5_data_dir_path = h5_data_dir_path
        self.json_data_dir_path = json_data_dir_path

    def read_h5(self):
        """
        This function reads all h5 data files and returns a DataFrame.
        """
        all_file_paths = []
        # Loop through all files and append all the h5 file paths to a list.
        for root, dirs, files in os.walk(self.h5_data_dir_path):
            for name in files:
                if '.h5' in name:
                    all_file_paths.append(os.path.join(root, name))
        # Find out all keys of each h5 file
        keys = pd.HDFStore(all_file_paths[0]).keys()
        frames = []
        for key_index in range(len(keys)):
            for file in all_file_paths:
                temp_frame = pd.read_hdf(file, key=keys[key_index])
                frames.append(temp_frame)
        h5_to_df = pd.concat(frames)
        return h5_to_df

    def read_json(self):
        """
        This function reads all json data files and returns a DataFrame.
        """
        all_file_paths = []
        # Loop through all files and append all the json file paths into a list.
        for root, dirs, files in os.walk(self.json_data_dir_path):
            for name in files:
                if '.json' in name:
                    all_file_paths.append(os.path.join(root, name))
        frames = []
        for file in all_file_paths:
            f = codecs.open(file)
            for line in f:
                line = line.strip("\n")
                temp_frame = json_normalize(json.loads(line))
                frames.append(temp_frame)
        json_to_df = pd.concat(frames)
        return self.add_tags_columns(json_to_df)

    @staticmethod
    def add_tags_columns(json_to_df):
        """
        This function returns a DataFrame with new added columns representing all tags.
        """
        df_with_tags = json_to_df
        # Loop through all rows corresponding to the column 'tags' of the DataFrame
        for row in range(len(df_with_tags['tags'])):
            # Loop through all items in each row
            for item_in_row in df_with_tags['tags'].iloc[row]:
                # If there is no column of this tag, then create a new column of it and fill in all zero values
                if str(item_in_row[0]) not in df_with_tags.columns:
                    df_with_tags[str(item_in_row[0])] = 0
                # Fill in the actual value (the number of people who tag the song)
                df_with_tags[str(item_in_row[0])].iloc[row] = item_in_row[1]
        return df_with_tags

    def merge_df(self):
        """
        This function returns a merged DataFrame based on 'track_id'.
        """
        return pd.merge(self.read_h5(), self.read_json(), on='track_id')

    def save_to_csv(self, csv_name):
        """
        This function saves the merged DataFrame to a csv file.
        """
        self.merge_df().to_csv(csv_name)


class DataClean(object):
    """
    This class performs several data cleaning functions, including removing unnecessary columns, dropping and filling
    missing values, deleting songs with no tags.
    """

    def __init__(self, dataset):
        """
        This is the constructor of the DataClean class.
        """
        self.data = dataset

    def drop_missing_values(self):
        """
        This function returns a slice of the original DataFrame with columns of all zeros deleted and columns of all
        missing values deleted.
        """
        self.data = self.data.loc[:, (self.data != 0).any(axis=0)].dropna(how='all', axis=1)

    def remove_columns(self):
        """
        This function returns a DataFrame with several columns deleted.
        """
        # This to_remove list contains track_id, artist_id, information about artist location (due to lots of blank
        # and not useful information), duplicated columns (title_y and artist_id), analysis_sample_rate (because the
        # values for this feature is the same for all instances)
        to_remove = ['analysis_sample_rate', 'audio_md5', 'track_id', 'track_7digitalid', 'artist_7digitalid',
                    'artist_id', 'release_7digitalid', 'artist_playmeid', 'artist_mbid', 'song_id', 'artist_longitude',
                    'artist_location', 'artist_latitude', 'title_y', 'artist', 'similars', 'timestamp']
        self.data.drop(set(to_remove) & set(self.data.columns), axis=1, inplace=True)

    def fill_missing_values(self):
        """
        This function returns a DataFrame with all missing values replace by 0.
        """
        self.data.fillna(0, inplace=True)

    def delete_songs(self):
        """
        This function returns a DataFrame with all songs which have no tags being deleted.
        """
        self.data = self.data[self.data['tags'] != '[]']

    def save_to_csv(self, csv_name):
        """
        This function saves the DataFrame to a csv file.
        """
        self.data.to_csv(csv_name)

