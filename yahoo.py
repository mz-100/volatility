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


class YFinanceData:
    """Class for loading options data from Yahoo Finance.

    Attributes:
        ticker (str): Ticker symbol.
        access_time (Timestamp): Date and time at which the data is loaded from Yahoo Finance.
        underlying_price (float): Current underlying asset.
        forward_prices (DataFrame): Pandas DataFrame containing the synthetic forward prices
            corresponding to different option maturities. The index of the DataFrame is the maturity
            date and the columns are:
                timeToMaturity (float): Time to maturity in years.
                discountFactor (float): Discount factor between access_time and this to maturity.
                forwardPrice (float): Forward price of the underlying asset for this maturity.
        options (DataFrame): Pandas DataFrame containing the options data. The index of the
            DataFrame is a multi-index with levels 'maturity' (date as a Timestamp), type
                ('P' or 'C' for put/call) and strike (float). The columns are:
                price (float): Option mid-price.
                spread (float): Bid-ask spread.
                volume (float): Traded volume.
                logStrike (float): log(strike/forward), i.e. the log-moneyness of a put option.
                OTM (bool): True if out-of-the-money or at-the-money, False if in-the-money.
                impliedVol (float): Implied volatility.
                totalVar (float): Total variance.
                delta (float): Delta.
            Note that not all the columns may be present (e.g. impliedVol and totalVar are added 
            after calling `process_data`).

    Methods:
        load_from_web: Loads options data from Yahoo Finance.
        process_data: Computes forward prices and implied volatilities.
        save_to_disk: Saves the object to disk in pickle or Excel format.
        load_from_disk: Loads the object from disk.

    Notes:
        Time in years is measured as the number of calendar days / 365. This needs to be changed in
        the future.
    """

    ticker: str
    access_time: pd.Timestamp
    underlying_price: float
    forward_prices: pd.DataFrame
    options: pd.DataFrame

    @staticmethod
    def load_from_web(ticker, min_maturity=None, max_maturity=None, expiration_time='16:00',
                      expiration_timezone='America/New_York', access_time=None):
        """Loads options data from Yahoo Finance.

        Args:
            ticker (str): Ticker symbol.
            min_maturity (str | float): Minimum maturity of the options to load. If a string, it
                must be in the format YYYY-MM-DD. If a float, must be time in years.
            max_maturity  (str | float): Maximum maturity of the options to load. If a string, it
                must be in the format YYYY-MM-DD. If a float, must be time in years.
            expiration_time (str): Time of day at which options expire. Must be in the format HH:MM.
            expiration_timezone (str) : Timezone of `expiration_time`.
            access_time (datetime): Time at which the data is accessed. If None, the current time is
                used. This is useful if you download data when the market is closed; in this case
                set `access_time` to the time at which the market closed. If `access_time` does not
                have timezone information, it is assumed to be in the same as `expiration_timezone`.

        Returns:
            YFinanceData object
        """
        # Empty object which we are going to fill with data
        data = YFinanceData()
        data.ticker = ticker

        yf_ticker = yf.Ticker(ticker)
        data.underlying_price = (yf_ticker.info['bid'] + yf_ticker.info['ask']) / 2

        # Set time at which the data is loaded.
        # The `.astimezone()` adds timezone information to the datetime object if it does not have
        # it (and does nothing if it already has it); the timezone is set to the system timezone by
        # default.
        # We need to be careful with timezone information when subtracting dates and times
        # (e.g. for computing time to maturity) for short options. In principle, for options with
        # not so short maturities this could be ignored.
        if access_time is None:
            data.access_time = pd.Timestamp(datetime.now().astimezone())
        else:
            data.access_time = pd.Timestamp(access_time.astimezone())

        expiration_tz = pytz.timezone(expiration_timezone)

        # Loading option data from Yahoo 
        # First we load each maturity to a separate DataFrame as returned by yfinance, then join
        options_dataframes = []

        # The following will be a dictionary mapping maturities as Pandas Timestamp to time to
        # maturities in years
        time_to_maturities = {}
        for maturity_str in yf_ticker.options:      # yfinance stores maturities as strings
            maturity = expiration_tz.localize(
                datetime.strptime(maturity_str + " " + expiration_time, "%Y-%m-%d %H:%M"))

            # Skip options with maturities out of the range specified by the user (if any)
            T = (maturity - data.access_time).total_seconds() / (365*24*60*60)
            if min_maturity is not None:
                if isinstance(min_maturity, str):
                    if datetime.strptime(min_maturity, "%Y-%m-%d") < maturity.date():
                        continue
                else:
                    if T < min_maturity:
                        continue
            if max_maturity is not None:
                if isinstance(max_maturity, str):
                    if datetime.strptime(max_maturity, "%Y-%m-%d") > maturity.date():
                        continue
                else:
                    if T > max_maturity:
                        continue

            time_to_maturities[pd.Timestamp(maturity.date())] = T

            # Get calls and puts tables for this maturity and join them
            calls = yf_ticker.option_chain(maturity_str).calls
            calls['type'] = 'C'
            puts = yf_ticker.option_chain(maturity_str).puts
            puts['type'] = 'P'
            calls_and_puts = pd.concat([calls, puts], ignore_index=True)
            calls_and_puts['maturity'] = pd.Timestamp(maturity.date())
            calls_and_puts['price'] = (calls_and_puts['bid'] + calls_and_puts['ask']) / 2
            calls_and_puts['spread'] = calls_and_puts['ask'] - calls_and_puts['bid']
            options_dataframes.append(calls_and_puts)

        # No options were loaded
        if not options_dataframes:
            raise RuntimeError('No options were loaded. '
                               'Check that the ticker and maturities are correct')

        # Make a single DataFrame which contains all the options. Keep only the columns we need.
        data.options = pd.concat(options_dataframes, ignore_index=True)[
                ['maturity', 'type', 'strike', 'price', 'spread', 'volume']]
        # Set multi-index and sort in the ascending order for maturities and strikes, and in the
        # descending order for type (i.e. 'P' before 'C' - this is convenient because when we
        # use only OTM options, puts will have smaller strikes, so we want them to be before calls)
        data.options.set_index(['maturity', 'type', 'strike'], inplace=True)
        data.options.sort_index(ascending=[True, False, True], inplace=True)

        # Make a DataFrame with forward prices, which will be filled later. Now it only contains
        # maturity and timeToMaturity columns
        data.forward_prices = pd.DataFrame(
            data=[(maturity, time_to_maturities[maturity])
                  for maturity in sorted(time_to_maturities)],
            columns=['maturity', 'timeToMaturity'])
        data.forward_prices.set_index('maturity', inplace=True)

        return data

    def process_data(self, discount_rate=0.0, otm_vol_only=True):
        """Computes forward prices and implied volatilities.

        Args:
            discount_rates (float | list): Risk-free rates to use for discounting. If a float,
                then a constant risk-free rate is assumed for all option maturities. If a list of
                tuples (t,r), then the first element of a tuple must be the time to maturity in
                years and the second element must be the corresponding risk-free rate. The risk-free
                rate is linearly interpolated between the points in the list and extrapolated flat.
            otm_vol_only (bool): If True, then implied volatilities are computed only for
                out-of-the-money and at-the-money options and set to NaN for in-the-money options.
                If False, then implied volatilities are computed for all options. Note: computation
                of implied volatility may fail for deep in-the-money options.

        Returns:
            The object itself with added forward prices and implied volatilities.

        Notes:
            The computation of forward prices is based on the call-put parity applied to the
            available market option prices. For each maturity, we use the pair of a call and a put
            with the same strike which have the smallest difference in price.
        """
        if not hasattr(self, 'options'):
            raise RuntimeError('You need to load options data first')
        
        #################################
        # Computation of forward prices #
        #################################

        # Make a function for computing the appropriate discount factors
        if np.isscalar(discount_rate):
            def discount(t):
                return math.exp(-t*discount_rate)
        else:
            discount_rate_interp = interpolate.interp1d(*zip(*discount_rate))

            def discount(t):
                if t <= discount_rate[0][0]:
                    r = discount_rate[0][1]
                elif t >= discount_rate[-1][0]:
                    r = discount_rate[-1][1]
                else:
                    r = discount_rate_interp(t)
                return math.exp(-r*t)

        # Fill in the discountRate column of forward_prices
        self.forward_prices['discountFactor'] = self.forward_prices.timeToMaturity.apply(discount)

        # A function for computing forward prices for each maturity, which will be called for each
        # row of the forward_prices DataFrame
        def compute_forward_price(forward):
            # maturity is the index of the forward_prices DataFrame, hence we use .name
            maturity = forward.name
            # Collect call and put options for the current maturity
            calls = self.options.loc[maturity].loc['C']
            puts = self.options.loc[maturity].loc['P']

            # By call-put parity, call and put prices must be equal for K=F. Hence as initial guess
            # for the forward price we take the strike where |C-P| is smallest
            # Note: we need to check for non-zero prices because some options may have zero prices
            # (e.g. if the volume is zero)
            price_diff = (calls.price[calls.price > 0] - puts.price[puts.price > 0]).dropna().abs()

            # Check that there are some options with non-zero prices and matching strikes
            # Typically, there will be no such options if we are trying to get data before the
            # market opens
            if not price_diff.empty:
                K = price_diff.idxmin()
                P = puts.loc[K].price
                C = calls.loc[K].price
                T = forward.timeToMaturity
                d = discount(T)
                return K + d*(C-P)
            else:
                return math.nan

        # Fill in the forwardPrice column of forward_prices
        self.forward_prices['forwardPrice'] = self.forward_prices.apply(compute_forward_price, axis=1)

        #######################################
        # Computation of implied volatilities #
        #######################################

        self.options['logStrike'] = self.options.apply(
            # index: x.name = (maturity, type, strike)
            lambda x: math.log(x.name[2] / self.forward_prices.loc[x.name[0]].forwardPrice),
            axis=1)
        self.options['OTM'] = self.options.apply(
            lambda x: x.logStrike >= 0 if x.name[1] == 'C' else x.logStrike <= 0,
            axis=1)

        # Define three functions for computing implied volatility, delta and total variance
        # which will be called for each row of the options DataFrame
        def compute_iv(option):     # option is a row of the options DataFrame
            if otm_vol_only and not option.OTM:
                return math.nan
            return implied_vol(
                forward_price=self.forward_prices.loc[option.name[0]].forwardPrice,
                maturity=self.forward_prices.loc[option.name[0]].timeToMaturity,
                strike=option.name[2],
                option_price=option.price,
                call_or_put_flag=(1 if option.name[1] == 'C' else -1),
                discount_factor=self.forward_prices.loc[option.name[0]].discountFactor)

        def compute_totalvar(option):
            if option.impliedVol is math.nan:
                return math.nan
            return option.impliedVol**2 * self.forward_prices.loc[option.name[0]].timeToMaturity
        
        def compute_delta(option):
            if option.impliedVol is math.nan:
                return math.nan
            d1 = 0.5*math.sqrt(option.totalVar) - option.logStrike / math.sqrt(option.totalVar)
            d = self.forward_prices.loc[option.name[0]].discountFactor
            T = self.forward_prices.loc[option.name[0]].timeToMaturity
            if option.name[1] == 'C':
                return d * norm.cdf(d1)
            else:
                return d * (norm.cdf(d1) - 1)

        self.options['impliedVol'] = self.options.apply(compute_iv, axis=1)
        self.options['totalVar'] = self.options.apply(compute_totalvar, axis=1)
        self.options['delta'] = self.options.apply(compute_delta, axis=1)

        return self

    def save_to_disk(self, filename, backend=None):
        """Saves data to disk.

        Supports pickle and Excel backends. The pickle backend saves the object as is. The Excel
        backend saves the object as an Excel file with three sheets: Info, Forwards and Options. The
        Info sheet contains `self.ticker`, `self.access_time` and `self.underlying_price` attributes
        of the object. The Forwards and Options sheets contain, respectively, `self.forward_prices`
        and 'self.options' DataFrames.

        Args:
            filename (str): Name of the file to save the object to.
            backend (str): Backend to use for saving the object. Must be 'pickle' or 'excel'. If
                None, the backend is determined from the filename extension: '.pkl' or '.dat' for
                pickle and  '.xlsx' for Excel.

        Returns:
            None
        """
        if not (hasattr(self, 'forward_prices') and hasattr(self, 'options')):
            raise RuntimeError('Looks like the data has not been loaded yet - aborting')
        if backend is None:
            # Determine backend from filename extension: pickle or excel
            if filename.endswith('.pkl') or filename.endswith('.dat'):
                backend = 'pickle'
            elif filename.endswith('.xlsx'):
                backend = 'excel'
            else:
                raise ValueError('Unknown file extension')

        if backend == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        elif backend == 'excel':
            # A DataFrame with one row for the Info sheet
            info = pd.DataFrame(
                data=[[self.ticker, self.access_time.strftime('%Y-%m-%d %H:%M:%S %z'),
                       self.underlying_price]],
                columns=['ticker', 'accessTime', 'underlyingPrice'])
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                info.to_excel(writer, sheet_name='Info', header=True, index=False)
                self.forward_prices.to_excel(writer, sheet_name='Forwards', header=True, index=True)
                self.options.to_excel(writer, sheet_name='Options', header=True, index=True)

            ##########################################
            # Nice visual formatiing of Excel sheets #
            ##########################################    
            W = load_workbook(filename)

            # Info sheet
            # Columns: A = ticker, B = accessTime, C = underlyingPrice
            W['Info'].column_dimensions['A'].width = len('ticker') + 2
            W['Info'].column_dimensions['B'].width = len('YYYY-MM-DD HH:MM:SS +HHMM') + 2
            W['Info'].column_dimensions['C'].width = len('underlyingPrice') + 2
            W['Info']['C2'].number_format = '0.0000'

            # Forwards sheet
            # Columns: A = maturity, B = timeToMaturity, C = discountRate, D = forwardPrice
            W['Forwards'].column_dimensions['A'].width = len('YYYY-MM-DD') + 2
            for cell in W['Forwards']['A']:
                cell.number_format = 'YYYY-MM-DD'
            for col in ['B', 'C', 'D']:
                W['Forwards'].column_dimensions[col].width = 16
                for cell in W['Forwards'][col]:
                    cell.number_format = '0.0000'

            # Options sheet
            # Columns: A = maturity, B = type, C = strike, D = price, E = spread,
            # F = volume, G = logStrike, H = OTM, I = impliedVol, J = totalVar, K = delta
            W['Options'].column_dimensions['A'].width = len('YYYY-MM-DD') + 2
            W['Options'].column_dimensions['B'].width = len('type') + 2
            W['Options'].column_dimensions['C'].width = len('strike') + 2
            for col in ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']:
                W['Options'].column_dimensions[col].width = 16
            for c in W['Options']['A']:
                c.number_format = 'YYYY-MM-DD'
            for col in ['D', 'E', 'F', 'G', 'I', 'J', 'K']:
                for cell in W['Options'][col]:
                    cell.number_format = '0.0000'

            # Finally freeze panes
            W["Forwards"].freeze_panes = "A2"
            W["Options"].freeze_panes = "D2"

            W.save(filename)
        else:
            raise ValueError(f'Unknown backend: {backend}')

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
            YFinanceData object
        """
        if backend is None:
            # Determine backend from filename extension: pickle or excel
            if filename.endswith('.pkl') or filename.endswith('.dat'):
                backend = 'pickle'
            elif filename.endswith('.xlsx'):
                backend = 'excel'
            else:
                raise ValueError('Unknown file extension')

        if backend == 'pickle':
            with open(filename, 'rb') as f:
                return pickle.load(f)
        elif backend == 'excel':
            data = YFinanceData()
            info = pd.read_excel(
                filename, sheet_name='Info', header=0, index_col=False)
            data.ticker = info.ticker[0]
            data.access_time = pd.Timestamp(
                datetime.strptime(info.accessTime[0], '%Y-%m-%d %H:%M:%S %z'))
            data.underlying_price = info.underlyingPrice[0]
            data.forward_prices = pd.read_excel(
                filename, sheet_name='Forwards', header=0, index_col=0)
            data.options = pd.read_excel(
                filename, sheet_name='Options', header=0, index_col=[0, 1, 2])
            return data
        else:
            raise ValueError(f'Unknown backend: {backend}')

    def get_maturities(self, min_maturity=None, max_maturity=None):
        """Returns the list of available maturities in the given range.

        Args:
            min_maturity (str | Timestamp): Return maturities not smaller than this value.
            max_maturity (str | Timestamp): Return maturities not greater than this value.

        Returns:
            A list of Timestamps representing the available maturities.
        """
        return self.forward_prices[min_maturity:max_maturity].index.to_list()
    
    def get_implied_vol(self, maturity, moneyness_bounds=None, delta_bounds=None,
                        return_total_var=False, as_numpy=False):
        """The implied volatility or total variance curve of OTM options at a given maturity.

        Args:
            maturity (str | datetime | Timestamp): Maturity of the options. If a string, the
                format must be accepted by Pandas.
            moneyness_bounds ((float, float)): Tuple of two floats representing the lower and upper
                bounds for the moneyness of the options to include in the curve. The moneyness is
                defined as forward/strike for calls and strike/forward for puts. Thus, OTM options
                have moneyness < 1, and ITM options have moneyness > 1. If moneyness_bounds is None,
                all options are included. To omit one of the bounds, set it to 0 or inf.
            delta_bounds ((float, float)): Tuple of two floats representing the lower and upper
                bounds for the delta of the options to include in the curve. If None, all options
                are included. To omit one of the bounds, set it to -1 or 1.
            return_total_var (bool): If False, return the implied volatility curve in the
                coordinates (strike, implied_vol). If True, return the total variance curve in the
                coordinates (log_strike, total_var), where `log_strike = log(strike/forward)`.
            as_numpy (bool): If True, then return the of two numpy arrays: (strikes, implied_vol) or
                (log_strike, total_var). If False, then return a Pandas Series with strikes or
                log-moneyness as the index and implied volatilities or total variances as values.

        Returns:
            A Pandas Series with strike as index and implied volatility as values (or log-moneyness
            and total variance) or a tuple of two numpy arrays, the first being the array of strikes
            and the second being the array of implied volatilities (or, again, log-moneyness and
            total variance).
        """
        if moneyness_bounds is None:
            moneyness_bounds = (-np.inf, np.inf)
        if delta_bounds is None:
            delta_bounds = (-np.inf, np.inf)

        # options with the given maturity and filtered for moneyness and delta
        options = self.options.loc[maturity].copy().reset_index()
        options['moneyness'] = options.apply(
            lambda option: math.exp(option.logStrike * (-1 if option.type == 'C' else 1)),
            axis=1)
        filtered_options = options[options.OTM &
                                   options.moneyness.between(*moneyness_bounds) &
                                   options.delta.between(*delta_bounds)]

        if return_total_var:
            returned_data = filtered_options.set_index('logStrike').sort_index().totalVar
        else:
            returned_data = filtered_options.set_index('strike').sort_index().impliedVol

        if as_numpy:
            return returned_data.index.to_numpy(), returned_data.to_numpy()
        else:
            return returned_data
