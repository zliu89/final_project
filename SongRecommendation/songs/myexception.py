'''
Created on Dec 4, 2016
@ProjectTitle: Song Recommendation
@author: Jingyi Su(js5991)， Qianyu Cheng(qc510), Luyu Jin(lj1035)
'''

class InvalidInputError(Exception):
    """
    This InvalidInputError class extends from the Extension class.
    """
    def __str__(self):
        return 'This is an invalid input. Please enter either value 0 or 1.'

