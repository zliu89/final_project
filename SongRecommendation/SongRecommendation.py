'''
Created on Dec 4, 2016
@ProjectTitle: Song Recommendation
@author: Jingyi Su(js5991)， Qianyu Cheng(qc510), Luyu Jin(lj1035)
'''

from songs import model
import sys
import pandas as pd
import warnings


"""
This is the main module for song recommendation. This module reads in featureSelectedData.csv, which is a
cleaned data with features selected. What this module do is to show the user 20 songs that are most popular,
so that user would at least know some of the songs. The user will respond to these songs with 0 as dislike and
1 as like. After that, our algorithm will recommend other songs to the users.
"""

if __name__ == '__main__':
    # Ignore warning messages.
    warnings.filterwarnings("ignore")

    # Catch Input/Output errors.
    try:
        print("Sounds loading...")
        # Read in "featureSelectedData.csv".
        data = pd.read_csv("featureSelectedData.csv", index_col=0,sep=",", encoding="ISO-8859-1")
    except IOError:
        # If IOError, print "cannot read the file"
        print("Cannot read the file")
        # Exit the system.
        sys.exit()

    # Print a description that explains how song recommendation system work.
    print("This is a concurrent song recommendation system from our music base. We are looking forward to hearing from you on whether you like the recommended song or not.\n"+"The system will be retrained after each song based on your preference. We strongly hope you stay in tune for the first few songs and your next favorite songs will be coming soon.")
    # Build a song class that recommends 20 songs to the user.
    songRecommend = model.Song(data,20)
    songRecommend.songRecommendationInitial()

    # Main program that recommends a new song to the users.
    while True:
        # Catch KeyboardInterrupt error and EOFErrors.
        try:
            print("Based on your prior selections, we recommend you with this new song.")
            # Keep recommending the songs.
            songRecommend.songRecommendation(1)
        except KeyboardInterrupt:
            sys.exit(1)
        except EOFError:
            print("EOFError")
            sys.exit(1)
    
