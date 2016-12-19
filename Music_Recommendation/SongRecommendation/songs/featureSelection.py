'''
Created on Dec 4, 2016
@ProjectTitle: Song Recommendation
@author: Jingyi Su(js5991)， Qianyu Cheng(qc510), Luyu Jin(lj1035)
'''
import pandas as pd
import sys


class FeatureSelection(object):
    """
    The class feature selection selects the features that are useful for song recommendation.
    """
    def __init__(self, dataSource):
        """
        :param dataSource: take in the original cleansed but without feature selection data.
        """
        # df1 is the data frame for attributes other than tags of songs. i.e. artist, titles and etc.
        self.df1 = dataSource.ix[:, :"tags"]

        # df2 is the data frame for attributes that only have tags. i.e. rocks, classics, and etc.
        self.df2 = dataSource.ix[:, "Bay Area":]

    def remove_tag_less_than_threshold(self, threshold=5):
        """
        This function removes tag columns that have less than threshold(default = 5) songs associated with them.
        :param threshold: default 5. Can be set to any number.
        :return: data frame that removed tags less than threshold.
        """
        # To standardize the numbers in tags, set up all number in tags as 1.
        self.df2 = self.df2.applymap(lambda x: 1 if x > 0 else 0)

        # Only keeps the tags that have larger than threshold of songs.
        self.df2 = self.df2[self.df2.columns[self.df2.sum() > threshold]]
    
    def return_data_set(self):
        """
        This function concatenates data frame 1 and data frame 2.
        :return: data frame that contains df1 and df2
        """
        return pd.concat([self.df1, self.df2], axis=1)
    
    def to_csv_file(self, directory):
        """
        :param directory: path to where csv file will be saved.
        :return: saved csv file.
        """
        self.return_data_set().to_csv(directory)
    
if __name__ == "__main__":
    # Catch exception for featureSelection main module.
    try:
        # Read in 'cleanedData.csv'.
        data = pd.read_csv("../cleanedData.csv", index_col=0, sep=",")

        # Build the FeatureSelection object: selectedData.
        selectedData = FeatureSelection(data)

        # Apply remove_tag_less_than_threshold() to selectedData.
        selectedData.remove_tag_less_than_threshold()

        # Save selectedData into a csv file.
        selectedData.to_csv_file("../featureSelectedData.csv")
        print("Saved to File at ../featureSelectedData.csv")
    except EOFError:
        sys.exit()
    except IOError:
        print("Cannot read/write file")
        sys.exit()
    except KeyboardInterrupt:
        print("User directed exit")
        sys.exit()
    
        
