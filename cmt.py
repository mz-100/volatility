import math
import numpy as np
import yfinance as yf
import pandas as pd
import pytz
import pickle
from scipy.stats import norm
from datetime import datetime
from scipy import interpolate
from openpyxl import load_workbook
from .impvol import implied_vol


class CMTData:
    """Class for loading the Treasury par yield curve (CMT rates) for a given date.

    Attributes:
        date: Date for which the data is loaded.
        par_yield (Series): Par yield rates as a Pandas time series object.

    Methods:
        load_from_web: Loads data from the Treasury website.
        save_to_disk: Saves the object to disk in pickle or Excel format.
        load_from_disk: Loads the object from disk.
        ytm_curve: Computes the yield-to-maturity curve from the par yield rates.
    """

    def __init__(self, date, par_yield):
        self.date = date
        self.par_yield = par_yield

    def load_from_web(self, date=None):
        """Loads the Treasury par yield curve from the Treasury website.

        Args:
            date (datetime): Date for which the data is loaded. If not provided, the current date is used.
        
        Returns:
            CMTData: The object itself with the loaded data.

        Notes:
            If the yield curve is not available for the requested date, the last prior available
            date is used.
        """
        pass

    def save_to_disk(self, filename, backend=None):
        """Saves data to disk.

        Args:
            filename (str): Name of the file.
            backend (str): Backend to use for saving the object. Must be 'pickle' or 'excel'. If
                None, the backend is determined from the filename extension: '.pkl' or '.dat' for
                pickle and  '.xlsx' for Excel.

        Returns:
            None
        """
        pass

    @staticmethod
    def load_from_disk(filename, backend=None):
        """Loads data from disk.

        Supports pickle and Excel backends. See `save_to_disk` for details on the format of the
        saved data.

        Warning: this method does not check that the provided file contains correct (and safe) data.
        Always perform safety check for data from untrusted sources.

        Args:
            filename (str): Name of the file to load the object from.
            backend (str): Backend to use for loading the object. Must be 'pickle' or 'excel'. If
                None, the backend is determined from the filename extension: '.pkl' or '.dat' for
                pickle and  '.xlsx' for Excel.

        Returns:
            CMTData object
        """
        pass

    def ytm_curve(self, as_numpy=False):
        """Computes the yield-to-maturity curve.

        Args:
            as_numpy (bool): If False, returns a Pandas Series. If True, returns two numpy arrays.

        Returns:
            A Pandas Series or a tuple of two numpy arrays, the first one containing the maturities
            and the second one containing the yield-to-maturity rates.
        """
        pass
    