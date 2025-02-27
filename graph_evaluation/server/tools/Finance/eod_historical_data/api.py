import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stock_price_data_api(symbol_name_exchange_id: str, to: str=None, fmt: str='json', period: str='d', is_from: str=None, order: str='a', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With End-of-Day data API, we have data for more than 150 000 tickers all around the world. We cover all US stocks, ETFs, and Mutual Funds (more than 51 000 in total) **from the beginning**, for example, the Ford Motors data is from Jun 1972 and so on. And non-US stock exchanges we cover mostly from Jan 3, 2000. We do provide daily, weekly and monthly data raw and adjusted to splits and dividends."
    symbol_name_exchange_id: consists of two parts: {SYMBOL_NAME}.{EXCHANGE_ID}, then you can use, for example, MCD.MX for Mexican Stock Exchange. or MCD.US for NYSE. Check the [list of supported exchanges](https://eodhistoricaldata.com/financial-apis/list-supported-exchanges/) to get more information about stock markets we do support.
        to: Date to.
        fmt: The output format. Possible values are ‘csv’ for CSV output and ‘json’ for JSON output. Default value: ‘csv’.
        period: Use ‘d’ for daily, ‘w’ for weekly, ‘m’ for monthly prices. By default, daily prices will be shown.
        is_from: Date from.
        order: Use ‘a’ for ascending dates (from old to new), ‘d’ for descending dates (from new to old). By default, dates are shown in ascending order.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/eod/{symbol_name_exchange_id}"
    querystring = {}
    if to:
        querystring['to'] = to
    if fmt:
        querystring['fmt'] = fmt
    if period:
        querystring['period'] = period
    if is_from:
        querystring['from'] = is_from
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def financial_news_api(limit: int=50, to: str='2020-12-30', offset: int=0, is_from: str='2020-01-01', s: str='AAPL.US', t: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Financial News API is available under all subscriptions. Each Financial News API request consumes 5 API calls. The Financial News API is a powerful tool that helps you get company news and filter out them by date, type of news and certain tickers with the given parameters."
    limit: The number of results should be returned with the query. Default value: 50, minimum value: 1, maximum value: 1000.
        offset: The offset of the data. Default value: 0, minimum value: 0. For example, to get 100 symbols starting from 200 you should use limit=100 and offset=200.
        s: REQUIRED if parameter ‘t’ not set. The ticker code to get news for.
        t: REQUIRED if parameter ‘s’ not set. The tag to get news on a given topic.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/news"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if to:
        querystring['to'] = to
    if offset:
        querystring['offset'] = offset
    if is_from:
        querystring['from'] = is_from
    if s:
        querystring['s'] = s
    if t:
        querystring['t'] = t
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def calendar_upcoming_earnings_trends_ipos_and_splits(type: str, to: str=None, symbols: str=None, fmt: str='json', is_from: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With our Financial Calendar data feed, we provide data about upcoming earnings, IPOs, and splits. If you are looking for an economic calendar, which includes an earnings calendar and IPOs calendar, this API is for you.
		
		To get access to Calendar API you should be subscribed either to Calendar API or to the ‘All-In-One’ data package, which includes all possible data feeds we have. More information with prices you can get on our main page.
		
		For IPOs we have dated from January 2015 and up to 2-3 weeks in the future. For splits, we have data from January 2015 up to several months in the future and full historical data is provided under our Splits and Dividends API. And for earnings, we have data from the beginning and up to several months in the future."
    type: 4 calendar functions:
**earnings, trends, ipos, splits**
        to: The end date for earnings data, if not provided, today + 7 days will be used.
        symbols: You can request specific symbols to get historical and upcoming data. If ‘symbols’ used, then ‘from’ and ‘to’ parameters will be ignored. You can use one symbol: ‘AAPL.US’ or several symbols separated by a comma: ‘AAPL.US, MS’. REQUIRED for Earnings Trends.
        fmt: Output format, possible values: ‘csv’ – for CSV output and ‘json’ – for JSON output. The data for trends is available only in JSON format due to a complex data structure. Default value is ‘csv’ for others.
        is_from: The start date for earnings data, if not provided, today will be used.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/calendar/{type}"
    querystring = {}
    if to:
        querystring['to'] = to
    if symbols:
        querystring['symbols'] = symbols
    if fmt:
        querystring['fmt'] = fmt
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bonds_fundamentals_and_historical_api(cusip_or_isin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We support US corporate bonds and Government Bonds in our database (for government bonds, see [Economic Data API](https://eodhistoricaldata.com/financial-apis/economic-data-api/)). There are always new corporate bonds on the market, if you didn’t find any particular bond, please contact us and we will add the data within 24 hours.
		Bonds fundamentals and historical data could be accessed either via ISIN or via CUSIP IDs. Other IDs are not supported at the moment."
    cusip_or_isin: CUSIP of a particular bond, it’s also could be an ISIN. Other IDs are not supported at the moment.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/bond-fundamentals/{cusip_or_isin}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def macroeconomics_data_and_macro_indicators_api(country: str, indicator: str='inflation_consumer_prices_annual', fmt: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Macroeconomics is a part of economics dealing with the performance, structure, behavior, and decision-making of an economy as a whole. Our Macroeconomics Data API includes regional, national, and global economies. We provide the data for more than 30 macro indicators such as GDP, unemployment rates, national income, price indices, inflation rates, consumption, international trades, and many other significant indicators.
		
		The Macroeconomics Data API is a part of Fundamental API and accessible under Fundamental subscription. Each Macroeconomics API request consumes 1 API call. To get macroeconomics indicators use the following URL:
		
		> https://eodhistoricaldata.com/api/macro-indicator/COUNTRY?api_token=YOUR_API_TOKEN&fmt=json&indicator=inflation_consumer_prices_annual
		
		**COUNTRY**: String. REQUIRED. Defines the country for which the indicator will be shown. The country should be defined in the Alpha-3 ISO format. Possible values: USA, FRA, DEU…
		**api_ token**: String. REQUIRED. Your api_token to access the API. You will get it after registration.
		**indicator**: String. OPTIONAL. Defines which macroeconomics data indicator will be shown. See the list of possible indicators below. The default value is ‘gdp_current_usd‘.
		**fmt**: String. OPTIONAL. The output format could be ‘json’ for JSON and ‘csv’ for CSV output. The default value is ‘json’."
    country: Defines the country for which the indicator will be shown. The country should be defined in the [Alpha-3 ISO format](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3).
        indicator: Defines which macroeconomics data indicator will be shown. See the list of possible indicators [here](https://eodhistoricaldata.com/financial-apis/macroeconomics-data-and-macro-indicators-api/). 
        fmt: The output format could be ‘json’ for JSON and ‘csv’ for CSV output. The default value is ‘json’.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/macro-indicator/{country}"
    querystring = {}
    if indicator:
        querystring['indicator'] = indicator
    if fmt:
        querystring['fmt'] = fmt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fundamental_data_api(symbol_name_exchange_id: str, filter: str='General::Code,General,Earnings', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simple access to fundamental data API for stocks, ETFs, Mutual Funds, and Indices from different exchanges and countries. Almost all major US, UK, EU, India, and Asia exchanges.
		
		Stocks, ETFs, Mutual Funds fundamental data
		
		Major US companies supported from 1985, more than 30 years and non-US symbols supported from 2000, it’s more than 21 years of the financial data. Symbols from major US exchanges (around 11000 tickers in total from NYSE, NASDAQ, and ARCA) 20 years both yearly and quarterly. For minor companies, we have data for the last 6 years and the previous 20 quarters. And the data is continually growing.
		We support more than 20.000 US Funds. Our database has equity funds as well as balanced and bond-based mutual funds.
		We also support details for more than 10,000 ETFs from different exchanges and countries.
		We provide Index Constituents (or Index Components) data for all major indices all around the world.
		Please note, not all companies report the whole financial data, then we can not guarantee that each company will have all data endpoints we do support.
		Due to a very complex data structure, we support fundamental data feeds only in JSON format."
    symbol_name_exchange_id: Consists of two parts: {SYMBOL_NAME}.{EXCHANGE_ID}, then you can use, for example, AAPL.MX for Mexican Stock Exchange. Or AAPL.US for NASDAQ.
        filter: The API supports field filtering with this parameter. We support multi-layer filtering. It’s also possible to use several, comma-separated, filters. For example: *General::Code,General,Earnings*
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/fundamentals/{symbol_name_exchange_id}"
    querystring = {}
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def intraday_historical_data_api(symbol_name_exchange_id: str, to: int=1564753200, interval: str='1h', fmt: str='json', is_from: int=1564752900, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Intraday Data API is available under ‘All World Extended’ and ‘All-In-One’ data packages. We support intraday historical data for major exchanges all around the world.
		- We have 1-minute intervals for US (NYSE and NASDAQ), including pre-market (premarket) and after-hours (afterhours) trading data from 2004, more than 15 years of the data. And 5-minute, 1-hour intervals from October 2020.
		- For Forex, Cryptocurrencies and MOEX tickers we have 1-minute intervals trading data from 2009, more than 12 years of the data. And 5-minute, 1-hour intervals from October 2020.
		- For other tickers. 5-minute, 1-hour intervals and only from October 2020.
		
		The data is updated 2-3 hours after market closing. For the US market, only NYSE and NASDAQ tickers are supported."
    symbol_name_exchange_id: Consists of two parts: {SYMBOLNAME}.{EXCHANGEID}, then you can use, for example, AAPL.MX for Mexican Stock Exchange. or AAPL.US for NASDAQ.
        to: Use these parameter to filter data by datetime. Parameters should be passed in UNIX time with UTC timezone, for example, these values are correct: “from=1564752900\\\\\\\\\\\\\\\" \\\\\\\\\\\\\\\"to=1564753200” and correspond to ‘ 2019-08-02 13:35:00 ‘ and ‘ 2019-08-02 13:40:00 ‘. The maximum periods between ‘from’ and ‘to’ are 120 days for 1-minute interval, 600 days for 5-minute interval and 7200 days for 1 hour interval. Default value is closest to now.
        interval: The possible intervals: ‘5m’ for 5-minutes, ‘1h’ for 1 hour, and ‘1m’ for 1-minute intervals. Default value is ‘5m’.
        fmt: Use this parameter to get the data in a different format. Possible values are ‘json’ for JSON and ‘csv’ for CSV. By default, the data is provided in CSV format.
        is_from: Use these parameters to filter data by datetime. Parameters should be passed in UNIX time with UTC timezone, for example, these values are correct: “from=1564752900&to=1564753200” and correspond to ‘ 2019-08-02 13:35:00 ‘ and ‘ 2019-08-02 13:40:00 ‘. The maximum periods between ‘from’ and ‘to’ are 120 days for 1-minute interval, 600 days for 5-minute interval and 7200 days for 1 hour interval. Default value is the maximum period till now.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/intraday/{symbol_name_exchange_id}"
    querystring = {}
    if to:
        querystring['to'] = to
    if interval:
        querystring['interval'] = interval
    if fmt:
        querystring['fmt'] = fmt
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def live_delayed_stock_prices_api(symbol_name_exchange_id: str, s: str='VTI,EUR.FOREX', filter: str='close', fmt: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We provide live (delayed) stock prices API for all subscribers of ‘All-World’, ‘All World Extended’, and ‘ALL-IN-ONE’ plans. With this API endpoint, you are able to get delayed (15-20 minutes) information about almost all stocks on the market.
		
		### Major features
		
		- We support **almost all symbols** and exchanges all around the world.
		- Prices are provided with **15-20 minutes delay**.
		- We provide only a **1-minute interval** within this API, then you will get prices only with 1-minute frequency.
		- **Multiple tickers** with one request.
		- **Supports Excel WEBSERVICE**."
    s: With this parameter you will be able to get data for multiple tickers at one request, all tickers should be separated with a comma.
We do not recommend using more than 15-20 tickers per request.
        filter: If you need only one field, just use this parameter.
For examples, if you use the following filter=close then only one number will be returned: 172.5. Which is very useful for Excel WEBSERVICE function like this:

`=WEBSERVICE(\"https://eodhistoricaldata.com/api/real-time/AAPL.US?api_token=OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX&fmt=json&filter=close\")`
        fmt: The output format. Possible values are ‘csv’ for CSV output and ‘json’ for JSON output. Default value: ‘csv’.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/real-time/{symbol_name_exchange_id}"
    querystring = {}
    if s:
        querystring['s'] = s
    if filter:
        querystring['filter'] = filter
    if fmt:
        querystring['fmt'] = fmt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stock_market_screener_api(signals: str=None, offset: int=0, limit: int=50, sort: str=None, filters: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Screener API is available under ‘All World Extended’ and ‘All-In-One’ data packages. Each Screener API request consumes 5 API calls. The Screener API is a powerful tool that helps you filter out tickers with the given parameters.
		
		The example of URL for the Screener API:
		
		> https://eodhistoricaldata.com/api/screener?api_token=YOUR_API_TOKEN&sort=market_capitalization.desc&filters=[["market_capitalization",">",1000],["name","match","apple"],["code","=","AAPL"],["exchange","=","us"],["sector","=","Technology"]]&limit=10&offset=0
		
		**filters**: String. OPTIONAL. Usage: filters=[[“field1”, “operation1”, value1],[“field2”, “operation2”, value2] , … ]. Filters out tickers by different fields.
		**signals**: String. OPTIONAL. Usage: signals=signal1,signal2,…,signalN. Filter out tickers by signals, the calculated fields.
		**sort**: String. OPTIONAL. Usage: sort=field_name.(asc|desc). Sorts all fields with type ‘Number’ in ascending/descending order.
		**api_token**: String. REQUIRED. Your api_token to access the API. You will get it after registration.
		**limit**: Number. OPTIONAL. The number of results should be returned with the query. Default value: 50, minimum value: 1, maximum value: 100.
		**offset**: Number. OPTIONAL. The offset of the data. Default value: 0, minimum value: 0, maximum value: 1000. For example, to get 100 symbols starting from 200 you should use limit=100 and offset=200.
		
		### List of Operations
		String operations are supported for all fields with type ‘String’. Numeric Operations are supported for all fields with type ‘NUMBER’:
		
		String Operations: [‘=’, ‘match’].
		Numeric Operations: [‘=’, ‘>’, ‘<‘, ‘>=’, ‘<=’, ‘!=’].
		
		Please note that each API request for Screener API consumes 5 API calls."
    signals: ### Filtering Data with Signals
You can use signals to filter tickers by different calculated fields. All signals are pre-calculated on our side. For example, if you need only tickers that have new lows for the past 200 days and the Book Value is negative, you can use the parameter ‘signal’ with the following value, to get all tickers with the criteria:

```
signals=bookvalue_neg,200d_new_lo

```
### List of supported Signals
```
50d_new_lo, 50d_new_hi, 200d_new_lo, 200d_new_hi – filters tickers that have new 50/200 days lows or new 50/200 days highs.
bookvalue_neg, bookvalue_pos – filters tickers with positive Book Value or with Negative Book Value.
wallstreet_lo, wallstreet_hi – filters tickers that have a price lower or higher than expected by Wall Street analysts.
```
        offset: The offset of the data. Default value: 0, minimum value: 0, maximum value: 1000. For example, to get 100 symbols starting from 200 you should use limit=100 and offset=200.
        limit: The number of results should be returned with the query. Default value: 50, minimum value: 1, maximum value: 100.
        sort: Usage: field_name.(asc|desc). Sorts all fields with type ‘Number’ in ascending/descending order.
        filters: ## Filtering data with Fields
You can use fields to filter the data. Fields have two types: Strings and Numbers. For strings should be used String Operations and for Numbers should be used Numeric Operations (see the chapter “List of Operations” in this documentation). For example, you can filter all companies with Market Capitalization above 1 billion, have only positive EPS within the ‘Personal Products’ industry, and with name started with the letter ‘B’.

`Usage: filters=[[“field1”, “operation1”, value1],[“field2”, “operation2”, value2] , … ]. Filters out tickers by different fields.`

### List of Supported Fields

- **code**: String. Filters by the ticker code.
- **name**: String. Filters by the ticker name.
- **exchange**: String. Filters by the exchange code. The list of all exchange codes is here.
- **sector**: String. Filters by sector. The list of sectors and industries is here.
- **industry**: String. Filters by industry. The list of sectors and industries is here.
- **market_capitalization**: Number. Filters by Market Capitalization, the latest value. Please note, that input for market_capitalization in USD.
- **earnings_share**: Number. Filters by Earnings-per-share (EPS), the latest value.
- **dividend_yield**: Number. Filters by Dividend yield, the latest value.
- **refund_1d_p**: Number. The last day gain/loss in percent. Useful to get top gainers, losers for the past day.
- **refund_5d_p**: Number. The last 5 days gain/loss in percent. Useful to get top gainers, losers for the past week.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/screener"
    querystring = {}
    if signals:
        querystring['signals'] = signals
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if sort:
        querystring['sort'] = sort
    if filters:
        querystring['filters'] = filters
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_exchanges(fmt: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We support more than 60 exchanges all around the world. All US exchanges are combined into one virtual exchange ‘US,’ which includes NYSE, NASDAQ, NYSE ARCA, and OTC/PINK tickers. All indices and commodities are in virtual exchanges INDX and COMM, respectively."
    fmt: The output format. Possible values are ‘csv’ for CSV output and ‘json’ for JSON output. Default value: ‘json’.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/exchanges-list/"
    querystring = {}
    if fmt:
        querystring['fmt'] = fmt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_exchange_details(exchange_code: str, to: str='2021-08-30', is_from: str='2021-05-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this API endpoint you will get detailed information about each exchange we do support, including:
		
		- Timezone – the timezone of exchange
		- isOpen – boolean value which indicates if exchange open right now or closed.
		- Trading hours and working days – open hours with working days for each exchange in the exchange timezone. This field could include also lunch hours if the exchange has it.
		- ActiveTickers – tickers with any activity for the past two months.
		- UpdatedTickers – tickers updated for the current day."
    to: The default value is 6 months after the current date.
        is_from: The default value is 6 months before the current date.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/exchange-details/{exchange_code}"
    querystring = {}
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_api(query_string: str, limit: int=15, bonds_only: int=0, type: str=None, exchange: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Our Search API for Stocks, ETFs, Mutual Funds, and Indices is one of the best ways to quickly search assets either by code or by company or asset name. The search engine automatically understands if there asset name or code or even ISIN and prioritizes the search fields accordingly. The search engine has several parameters for result ordering. We take into account not only search queries but also market capitalization and the average trading volume for the past period."
    limit: The number of results should be returned with the query. Default value: 15. If the limit is higher than 50, it will be automatically reset to 50.
        bonds_only: The default value is 0 and search returns only tickers, ETFs, and funds. To get bonds in result use value 1.
        type: The default value is ‘all’. You can specify the type of asset you search for.

- Possible values: all, stock, etf, fund, bonds, index, commodity, crypto.
- Please note: with the value ‘all’ bonds will not be displayed, you should explicitly request bonds.
        exchange: Filters output by exchange. Allowed input is the exchange code, for example: US, PA, CC, FOREX and others. In addition, it’s possible to use ‘NYSE’ and ‘NASDAQ’ exchange codes to filter out only tickers from these exchanges.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/search/{query_string}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if bonds_only:
        querystring['bonds_only'] = bonds_only
    if type:
        querystring['type'] = type
    if exchange:
        querystring['exchange'] = exchange
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bulk_api_for_eod_splits_and_dividends(exchange_id: str, symbols: str='MSFT, AAPL', fmt: str='json', date: str=None, filter: str=None, type: str='splits', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API allows to easily download the data for the **entire exchange** for a particular day. It works for end-of-day historical data feed as well as for splits and dividends data. For US tickers you can also use NYSE, NASDAQ, BATS, or AMEX as exchange symbols to get data only for NYSE or only for NASDAQ exchange.
		
		With this entire stock market API endpoint, you need not perform thousands and thousands of API requests per day. It’s not necessary anymore. We developed a bulk download API endpoint, and it’s easy to **download historical data for any day in bulk**."
    symbols: To download last day data for several symbols, for example, for MSFT and AAPL, you can add the ‘symbols’ parameter. For non-US tickers, you should use the exchange code, for example, BMW.XETRA or SAP.F
        fmt: The output format. Possible values are ‘csv’ for CSV output and ‘json’ for JSON output. Default value: ‘csv’.
        date: By default, the data for last trading day will be downloaded, but if you need any specific date, add ‘date’ parameter to the URL, in the following example we used September 21, 2017
        filter: If you need more data, like company name, you can use this parameter and get an extended dataset, which includes company name, EMA 50 and EMA 200 and average volumes for 14, 50 and 200 days.


        type: General bulk (batch) API for EOD, Splits, and Dividends


        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/eod-bulk-last-day/{exchange_id}"
    querystring = {}
    if symbols:
        querystring['symbols'] = symbols
    if fmt:
        querystring['fmt'] = fmt
    if date:
        querystring['date'] = date
    if filter:
        querystring['filter'] = filter
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def insider_transactions_api(code: str='AAPL.US', to: str=None, is_from: str=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The insider transactions API data is available for all US companies that report Form 4 to SEC. Insider trading involves trading in a public company’s stock by someone who has non-public, material information about that stock for any reason. In some cases, insider transactions could be very useful for making investment decisions.
		
		The Insider Transactions Data API is a part of Fundamental API and accessible under Fundamental subscription. Each Insider Transactions API request consumes 1 API call."
    code: To get the data only for Apple Inc (AAPL), use AAPL.US or AAPL ticker code. By default, all possible symbols will be displayed.
        to: Date to. Default value – the current date.
        is_from: Date from. Default value – one year ago.
        limit: The limit for entries per result, from 1 to 1000. Default value: 100.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/insider-transactions"
    querystring = {}
    if code:
        querystring['code'] = code
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def technical_indicator_api(symbol_name_exchange_id: str, function: str, is_from: str='2017-08-01', to: str='2020-01-01', fmt: str='json', order: str='d', splitadjusted_only: int=None, period: int=50, slow_kperiod: int=None, fast_kperiod: int=None, acceleration: int=None, signal_period: int=None, fast_period: int=None, filter: str=None, agg_period: str=None, slow_period: int=None, maximum: int=None, fast_dperiod: int=None, slow_dperiod: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Technical Indicator API is available under ‘All World Extended’ and ‘All-In-One’ data packages. Each Technical API request consumes 5 API calls."
    symbol_name_exchange_id: Consists of two parts: {SYMBOLNAME}.{EXCHANGEID}, then you can use, for example, AAPL.MX for Mexican Stock Exchange. or AAPL.US for NASDAQ.
        function: ## Technical Indicator Functions
For all functions you can use the following parameters: to, from, order and fmt. In addition, you should use function parameter, we described the specific usage for each function [here](https://eodhistoricaldata.com/financial-apis/technical-indicators-api/).

- Split Adjusted Data - **splitadjusted**
- Average Volume - **avgvol**
- Average Volume by Price - **avgvolccy**
- SMA - **sma**
- EMA - **ema**
- WMA - **wma**
- Volatility - **volatility**
- Stochastic Technical Indicator - **stochastic**
- Relative Strength Index - **rsi**
- Standard Deviation - **stddev**
- Stochastic Relative Strength Index - **stochrsi**
- Slope (Linear Regression) - **slope**
- Directional Movement Index - **dmi**
- Average Directional Movement Index - **adx**
- Moving Average Convergence/Divergence - **macd**
- Average True Range - **atr**
- Commodity Channel Index - **cci**
- Parabolic SAR - **sar**
- Bollinger Bands - **bbands**
- Amibroker File format - **format_amibroker**
        is_from: You can use this parameter with format ‘YYYY-MM-DD’.
        to: You can use this parameter with format ‘YYYY-MM-DD’.
        fmt: The output format, could be ‘json’ for JSON and ‘csv’ for CSV output. The default value is ‘json’.
        order: Use ‘a’ for ascending dates (from old to new) and ‘d’ for descending dates (from new to old). By default, dates are shown in ascending order.
        splitadjusted_only: Default value is ‘0’. By default, we calculate data for some functions by closes adjusted with splits and dividends. If you need to calculate the data by closes adjusted only with splits, set this parameter to ‘1’. Works with the following functions: sma, ema, wma, volatility, rsi, slope, and macd.
        period: The number of data points used to calculate each moving average value. Valid range from 2 to 100000 with the default value – 50.
        slow_kperiod: Slow K-period, the default value is 3. Valid range from 2 to 100000. (for Stochastic Technical Indicator)
        fast_kperiod: Fast K-period, the default value is 14. Valid range from 2 to 100000. (for Stochastic Technical Indicator, Stochastic Relative Strength Index)
        acceleration: Acceleration Factor used up to the Maximum value. Default value – 0.02. (for Parabolic SAR)
        signal_period: For Moving Average Convergence/Divergence
        fast_period: For Moving Average Convergence/Divergence
        filter: We also support the ability to get only the last value.
        agg_period: Aggregation period. Default value – ‘d’. Possible values: d – daily, w – weekly, m – monthly. (for Split Adjusted Data)
        slow_period: For Moving Average Convergence/Divergence
        maximum: Acceleration Factor Maximum value. Default value – 0.20. (for Parabolic SAR)
        fast_dperiod: Fast D-period, the default value is 14. Valid range from 2 to 100000. (for Stochastic Relative Strength Index)
        slow_dperiod: Slow D-period, the default value is 3. Valid range from 2 to 100000. (for Stochastic Technical Indicator)
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/technical/{symbol_name_exchange_id}"
    querystring = {'function': function, }
    if is_from:
        querystring['from'] = is_from
    if to:
        querystring['to'] = to
    if fmt:
        querystring['fmt'] = fmt
    if order:
        querystring['order'] = order
    if splitadjusted_only:
        querystring['splitadjusted_only'] = splitadjusted_only
    if period:
        querystring['period'] = period
    if slow_kperiod:
        querystring['slow_kperiod'] = slow_kperiod
    if fast_kperiod:
        querystring['fast_kperiod'] = fast_kperiod
    if acceleration:
        querystring['acceleration'] = acceleration
    if signal_period:
        querystring['signal_period'] = signal_period
    if fast_period:
        querystring['fast_period'] = fast_period
    if filter:
        querystring['filter'] = filter
    if agg_period:
        querystring['agg_period'] = agg_period
    if slow_period:
        querystring['slow_period'] = slow_period
    if maximum:
        querystring['maximum'] = maximum
    if fast_dperiod:
        querystring['fast_dperiod'] = fast_dperiod
    if slow_dperiod:
        querystring['slow_dperiod'] = slow_dperiod
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def options_data_api(symbol: str, trade_date_to: str=None, to: str=None, is_from: str=None, contract_name: str=None, trade_date_from: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We provide stock options data for top US stocks from NYSE and NASDAQ, the data for Options starts with April 2018. Options data is updated on a daily basis, however, we do not provide a history for options contracts’ prices or other data. That means: for each contract, there is only the current price, bid/ask, etc. For example, for AAPL today (May 7th, 2021) we have 2439 PUT and CALL option contracts in our database."
    symbol: Could be any supported symbol. No default value.
        trade_date_to: Filters OPTIONS by Last Trade Date Time. Default value: NONE.
        to: Filters OPTIONS by Expiration Date. Default value: '2100-01-01'.
        is_from: Filters OPTIONS by Expiration Date. Default value: today.
        contract_name: Returns only the data for particular contract.
        trade_date_from: Filters OPTIONS by Last Trade Date Time. Default value: NONE.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/options/{symbol}"
    querystring = {}
    if trade_date_to:
        querystring['trade_date_to'] = trade_date_to
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    if contract_name:
        querystring['contract_name'] = contract_name
    if trade_date_from:
        querystring['trade_date_from'] = trade_date_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_dividends_api(symbol_name_exchange_id: str, to: str=None, is_from: str=None, fmt: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get dividends for any ticker."
    symbol_name_exchange_id: Consists of two parts: {SYMBOLNAME}.{EXCHANGEID}, then you can use, for example, AAPL.MX for Mexican Stock Exchange. Or AAPL.US for NASDAQ.
        to: Date to. Default value is the closest available date to now.
        is_from: Date from. Default value is the earliest available date.
        fmt: Use this parameter to get the data in a different format. Possible values are ‘json’ for JSON and ‘csv’ for CSV. By default, the data is provided in CSV format.
Please note, that the extended format with declaration date, record date, and the payment date is available only for major US tickers and only in JSON format.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/div/{symbol_name_exchange_id}"
    querystring = {}
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    if fmt:
        querystring['fmt'] = fmt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_splits_api(symbol_name_exchange_id: str, is_from: str=None, fmt: str='json', to: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To get splits for any tickers."
    symbol_name_exchange_id: Consists of two parts: {SYMBOLNAME}.{EXCHANGEID}, then you can use, for example, AAPL.MX for Mexican Stock Exchange. Or AAPL.US for NASDAQ.
        fmt: Use this parameter to get the data in a different format. Possible values are ‘json’ for JSON and ‘csv’ for CSV. By default, the data is provided in CSV format.
        to: Date to. Default value is the closest available date to now.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/splits/{symbol_name_exchange_id}"
    querystring = {}
    if is_from:
        querystring['from'] = is_from
    if fmt:
        querystring['fmt'] = fmt
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_tickers(exchange_code: str, fmt: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "We support more than 60 exchanges all around the world. All US exchanges are combined into one virtual exchange ‘US,’ which includes NYSE, NASDAQ, NYSE ARCA, and OTC/PINK tickers. All indices and commodities are in virtual exchanges INDX and COMM, respectively.
		
		To get the full list of supported exchanges with names, codes, operating MICs, country, and currency, you can use the ‘exchanges-list’ endpoint"
    fmt: The output format. Possible values are ‘csv’ for CSV output and ‘json’ for JSON output. Default value: ‘csv’.
        
    """
    url = f"https://eod-historical-data.p.rapidapi.com/exchange-symbol-list/{exchange_code}"
    querystring = {}
    if fmt:
        querystring['fmt'] = fmt
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "eod-historical-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

