#!/usr/bin/env python3

"""Preprocess the network flow files.

The preprocessing involves extracting volume information (bytes and packets)
and nominal information (entropy of IPs and Ports). The entropy is extracted
using the classical Shannons Entropy formula.
"""

from math import sqrt, log
import pandas as pd
import numpy as np


class Preprocessor():
    """Class for preprocessor."""

    def __init__(self, filename):
        """Constructor, takes the name of the file to extract data from."""
        try:
            f = pd.read_csv(filename, parse_dates=True, index_col=0)
            splitted = list(f.groupby(pd.TimeGrouper('1Min')))
            timestamps = [name for name, group in splitted]
            bps = [group['bytes'].sum()/60 for name, group in splitted]
            print('bps')
            pps = [group['packets'].sum()/60 for name, group in splitted]
            sa_en = [self.shannon(group['src_addr'].value_counts())
                     for name, group in splitted]
            da_en = [self.shannon(group['dst_addr'].value_counts())
                     for name, group in splitted]
            sp_en = [self.shannon(group['src_port'].value_counts())
                     for name, group in splitted]
            dp_en = [self.shannon(group['dst_port'].value_counts())
                     for name, group in splitted]

            self.extracted = {
                'bps':   pd.Series(bps, index=timestamps),
                'pps':   pd.Series(pps, index=timestamps),
                'sa_en': pd.Series(sa_en, index=timestamps),
                'da_en': pd.Series(da_en, index=timestamps),
                'sp_en': pd.Series(sp_en, index=timestamps),
                'dp_en': pd.Series(dp_en, index=timestamps)
            }
        except:
            print('File not found.')

    def shannon(self, observations):
        """Given a set of observations calculates Shannons entropy."""
        sigma = observations.sum()
        entropy = 0
        for p in observations:
            entropy += (p / sigma) * log(p / sigma, 2)
        return -entropy

    def extract(self, attribute):
        """Returns the preprocessing of a single attribute."""
        try:
            return self.extracted[attribute]
        except:
            print('Attribute not found.')

    def get_all(self):
        """Returns all attributes in a pandas dataframe."""
        return self.extracted
