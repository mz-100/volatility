import math
import numpy as np
import pandas as pd
import datetime 
from scipy.optimize import root_scalar
from openpyxl import load_workbook


class CMTData:
    """Class for loading the Treasury par yield curve (CMT rates) for a given date.

    Attributes:
        date (date): Date for which the data is loaded.
        maturities (array): Available maturities, counted in years after `date`. Currently, the
            maturities are hard-coded to 1 Mo, 3 Mo, 6 Mo, 1 Yr, 2 Yr, 3 Yr, 5 Yr, 7 Yr, 10 Yr,
            20 Yr, and 30 Yr.
        yield (array): Par yield rates for the available maturities, as a decimal.

    Methods:
        load_from_web: Loads data from the Treasury website.
        save_to_disk: Saves data to disk.
        load_from_disk: Loads data from disk.
        ytm: Computes the yield-to-maturity curve from the par yield curve.
    """

    date: datetime.date
    maturities: np.array
    par_yield: np.array

    @staticmethod
    def load_from_web(date=None):
        """Loads the Treasury par yield curve from the Treasury website.

        Args:
            date (str): Date for which the data is loaded in the YYYY-MM-DD format. If not provided,
            the current date is used.
        
        Returns:
            CMTData: The object itself with the loaded data.

        Notes:
            If the yield curve is not available for the requested date, the nearest available date
            is used.
        """
        data = CMTData()
        data.maturities = np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])

        if date is None:
            date_ = datetime.datetime.now().date()
        else:
            date_ = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        
        year = date_.year
        url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
        df = pd.read_csv(url, parse_dates=['Date'], index_col='Date', date_format='%m/%d/%Y',
                         usecols=['Date', '1 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr',
                                  '7 Yr', '10 Yr', '20 Yr', '30 Yr'])
        i = df.index.get_indexer([pd.Timestamp(date_)], method='nearest')[0]
        data.date = df.index[i].date()
        data.par_yield = df.iloc[i].to_numpy()/100

        return data

    def save_to_disk(self, filename):
        """Saves data to disk in Excel format.

        Args:
            filename (str): Name of the file.

        Returns:
            None

        Notes:
            If the file exists, a sheet called 'CMT' will be added to it.
        """
        cmt = pd.DataFrame(
            data=[self.par_yield*100],
            index=[self.date],
            columns=['1 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr',
                    '20 Yr', '30 Yr'])
        cmt.index.rename('Date', inplace=True)

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            cmt.to_excel(writer, sheet_name='CMT', header=True, index=True)

        # Nice formatting
        W = load_workbook(filename)
        W['CMT'].column_dimensions['A'].width = len('YYYY-MM-DD') + 2
        for col in'BCDEFGHIJKL': 
            W['CMT'].column_dimensions[col].width = 7
        W.save(filename)

    @staticmethod
    def load_from_disk(filename):
        """Loads data from disk.

        Args:
            filename (str): Name of the file.

        Returns:
            CMTData object.
        """
        data = CMTData()
        df = pd.read_excel(filename, sheet_name='CMT', header=0, index_col='Date', 
                           usecols=['Date', '1 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr',
                                    '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr'])
        data.date = df.index[0].date()
        data.maturities = np.array([1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
        data.par_yield = df.iloc[0].to_numpy()/100
        return data

    def ytm(self):
        """Computes the yield-to-maturity curve.

        Returns:
            An array of the same shape as `self.maturities` with yield-to-maturity for each
            available maturity.
        """
        ytm = np.empty_like(self.par_yield)

        # For 1, 3, 6 months we can ignore the coupons
        ytm[0] = 12*math.log(1+self.par_yield[0]/12)   # 1 month
        ytm[1] = 4*math.log(1+self.par_yield[1]/4)     # 3 months
        ytm[2] = 2*math.log(1+self.par_yield[2]/2)     # 6 months

        # Bootstrap the rest
        for i in range(3, len(self.maturities)):
            t = self.maturities[i]
            tprev = self.maturities[i-1]
            coupon = self.par_yield[i]/2
            # The next formula is due to the replication argument.
            # By definition of par yield, the price of a bond with par yield `r` and face value 1
            # is 1. 
            # Then we try to replicate the bond maturiting at `t` with the bond maturing at `tprev`
            # (they pay the same coupons until `tprev`) minus the discounted face value of the 
            # latter bond, and discounted coupons paid between `tprev` and `t`.
            price = lambda r: (
                1/self.par_yield[i-1]*2*coupon*(1 - math.exp(-ytm[i-1]*tprev)) 
                + math.exp(-r*t) 
                + coupon*sum(math.exp(-r*s) for s in np.arange(tprev+0.5, t+0.1, 0.5)) 
                - 1)
            res = root_scalar(f=price, method='Newton', x0=ytm[i-1])
            if res.converged:
                ytm[i] = res.root
            else:
                ytm[i] = math.nan

        return ytm
    