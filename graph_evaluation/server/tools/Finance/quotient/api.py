import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def equity_dividends(symbol: str, to: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return dividends history data for a given security."
    symbol: The symbol of the asset to look for, e.g., `AAPL` (Apple Inc), `T` (AT&T Inc).
        to: The query end date (supported format is : **YYYY-mm-dd**), e.g., `2020-04-17`.
        from: The query start date (supported format is : **YYYY-mm-dd**), e.g., `2019-01-01`.
        
    """
    url = f"https://quotient.p.rapidapi.com/equity/dividends"
    querystring = {'symbol': symbol, 'to': to, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_symbol(query: str, categories: str, regions: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup for a Symbol or Name."
    query: Partial Company Name or Symbol, e.g., `apple`, `qualcomm`.
        categories: Asset categories. Supported categories are : [`EQT, IND, ETF, FUNDS, FX, CRYPTO`]. They can be mixed separated by a comma , e.g., `EQT,ETF`.
        regions: The region(s) in which to look for. Supported regions are : [`US, CA, UK, EU, AU`]. They can be mixed separated by a comma. They can be mixed separated by a comma , e.g., `US,UK,EU`.
        
    """
    url = f"https://quotient.p.rapidapi.com/search/symbol"
    querystring = {'query': query, 'categories': categories, 'regions': regions, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crypto_daily(is_from: str, to: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return end of day (daily) time series (Date, Open, High, Low, Close, Volume) given the input parameters."
    from: The query start date (supported format is : **YYYY-mm-dd**), e.g., `2018-04-01`.
        to: The query end date (supported format is : **YYYY-mm-dd**), e.g., `2020-04-21`.
        symbol: The symbol of the cryptocurrency pair to look for, e.g., `BTC/USD` (Bitcoin USD), `BTC-EUR` (Bitcoin EUR), `ETH/BTC` or `ETH-BTC` or `ETHBTC` (Ethereum BTC).
        
    """
    url = f"https://quotient.p.rapidapi.com/crypto/daily"
    querystring = {'from': is_from, 'to': to, 'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crypto_intraday(interval: int, to: str, symbol: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return intraday time series (Date, Open, High, Low, Close, Volume) given the input parameters."
    interval: 1-minute level time interval, e.g., `1` (1 min), `5` (5 min).
        to: The query end date (supported formats are : **YYYY-mm-dd HH:MM**, **YYYY-mm-dd**), e.g., `2020-04-21 10:30` or simply `2020-04-22`.
        symbol: The symbol of the cryptocurrency pair to look for, e.g., `BTC/USD` (Bitcoin USD), `BTC-EUR` (Bitcoin EUR), `ETH/BTC` or `ETH-BTC` or `ETHBTC` (Ethereum BTC).
        from: The query start date (supported formats are : **YYYY-mm-dd HH:MM**, **YYYY-mm-dd**), e.g., `2020-04-21 10:00` or simply `2020-04-21`.
        
    """
    url = f"https://quotient.p.rapidapi.com/crypto/intraday"
    querystring = {'interval': interval, 'to': to, 'symbol': symbol, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crypto_live(symbol: str, timezone: str='UTC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return current market price data given the input parameters."
    symbol: The symbol of the cryptocurrency pair to look for, e.g., `BTC/USD` (Bitcoin USD), `BTC-EUR` (Bitcoin EUR), `ETH/BTC` or `ETH-BTC` or `ETHBTC` (Ethereum BTC).
        timezone: Alternatively, a valid time zone for the returned timestamp, e.g., `US/Eastern`.
        
    """
    url = f"https://quotient.p.rapidapi.com/crypto/live"
    querystring = {'symbol': symbol, }
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def crypto_signal(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a trading signal based on market sentiment, reliable indicators, analyst ratings and news. Signal : `1`=**buy**, `0`=**hold** and `-1`=**sell**. Confidence interval : [0%-100%]."
    symbol: The symbol of the cryptocurrency pair to look for, e.g., `BTC/USD` (Bitcoin USD), `BTC-EUR` (Bitcoin EUR), `ETH/BTC` or `ETH-BTC` or `ETHBTC` (Ethereum BTC).
        
    """
    url = f"https://quotient.p.rapidapi.com/crypto/signal"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forex_live(symbol: str, timezone: str='UTC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return current market price data given the input parameters."
    symbol: The symbol of the currency pair to look for, e.g., `EUR/USD` or `EUR-USD` or `EURUSD`.
        timezone: Alternatively, a valid time zone for the returned timestamp, e.g., `US/Eastern`.
        
    """
    url = f"https://quotient.p.rapidapi.com/forex/live"
    querystring = {'symbol': symbol, }
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forex_daily(is_from: str, symbol: str, to: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return end of day (daily) time series (Date, Open, High, Low, Close, Volume) given the input parameters."
    from: The query start date (supported format is : **YYYY-mm-dd**), e.g., `2018-04-01`.
        symbol: The symbol of the currency pair to look for, e.g., `EUR/USD` or `EUR-USD` or `EURUSD`.
        to: The query end date (supported format is : **YYYY-mm-dd**), e.g., `2020-04-21`.
        
    """
    url = f"https://quotient.p.rapidapi.com/forex/daily"
    querystring = {'from': is_from, 'symbol': symbol, 'to': to, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def indexes_signal(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a trading signal based on market sentiment, reliable indicators, analyst ratings and news. Signal : `1`=**buy**, `0`=**hold** and `-1`=**sell**. Confidence interval : [0%-100%]."
    symbol: The symbol of the index to look for, e.g., `^GSPC` (S&P 500 index) or with suffix notation `SPX:INDEX` (S&P 500 index), `^GSPTSE` (TSX Composite Index) or with suffix notation `TXCX:INDEXCA` (TSX Composite Index). Valid suffixes are :

  - `:INDEX`: for world indices, e.g., `SPX:INDEX` (S&P 500 index) or `^GSPC`
  - `:INDEXUS`: for us specific indices, e.g., `SREN:INDEXUS` (S&P 500 Energy (Sector)) or `^GSPE`
  - `:INDEXCA`: for canadian specific indices, e.g., `TXCX:INDEXCA` (TSX Composite Index) or `^GSPTSE`
  - `:INDEXAU`: for australian specific indices, e.g., `XTO:INDEXAU` (ASX 100 Index) or `^ATOI`
  - `:INDEXEU`: for european specific indices, e.g., `BEL2I:INDEXEU` (BEL 20 Gr) or `BEL2I.BR`

  Please use the lookup endpoint to find out the symbol you're looking for.

        
    """
    url = f"https://quotient.p.rapidapi.com/indexes/signal"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def indexes_daily(is_from: str, to: str, symbol: str, adjust: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return end of day (daily) time series (Date, Open, High, Low, Close, Volume) given the input parameters."
    from: The query start date (supported format is : **YYYY-mm-dd**), e.g., `2018-04-01`.
        to: The query end date (supported format is : **YYYY-mm-dd**), e.g., `2020-04-21`.
        symbol: The symbol of the index to look for, e.g., `^GSPC` (S&P 500 index) or with suffix notation `SPX:INDEX` (S&P 500 index), `^GSPTSE` (TSX Composite Index) or with suffix notation `TXCX:INDEXCA` (TSX Composite Index). Valid suffixes are :

  - `:INDEX`: for world indices, e.g., `SPX:INDEX` (S&P 500 index) or `^GSPC`
  - `:INDEXUS`: for us specific indices, e.g., `SREN:INDEXUS` (S&P 500 Energy (Sector)) or `^GSPE`
  - `:INDEXCA`: for canadian specific indices, e.g., `TXCX:INDEXCA` (TSX Composite Index) or `^GSPTSE`
  - `:INDEXAU`: for australian specific indices, e.g., `XTO:INDEXAU` (ASX 100 Index) or `^ATOI`
  - `:INDEXEU`: for european specific indices, e.g., `BEL2I:INDEXEU` (BEL 20 Gr) or `BEL2I.BR`

  Please use the lookup endpoint to find out the symbol you're looking for.

        adjust: Tell if price need to be adjusted.
        
    """
    url = f"https://quotient.p.rapidapi.com/indexes/daily"
    querystring = {'from': is_from, 'to': to, 'symbol': symbol, }
    if adjust:
        querystring['adjust'] = adjust
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def indexes_live(symbol: str, timezone: str='UTC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return current market price data given the input parameters."
    symbol: The symbol of the index to look for, e.g., `^GSPC` (S&P 500 index) or with suffix notation `SPX:INDEX` (S&P 500 index), `^GSPTSE` (TSX Composite Index) or with suffix notation `TXCX:INDEXCA` (TSX Composite Index). Valid suffixes are :

  - `:INDEX`: for world indices, e.g., `SPX:INDEX` (S&P 500 index) or `^GSPC`
  - `:INDEXUS`: for us specific indices, e.g., `SREN:INDEXUS` (S&P 500 Energy (Sector)) or `^GSPE`
  - `:INDEXCA`: for canadian specific indices, e.g., `TXCX:INDEXCA` (TSX Composite Index) or `^GSPTSE`
  - `:INDEXAU`: for australian specific indices, e.g., `XTO:INDEXAU` (ASX 100 Index) or `^ATOI`
  - `:INDEXEU`: for european specific indices, e.g., `BEL2I:INDEXEU` (BEL 20 Gr) or `BEL2I.BR`

  Please use the lookup endpoint to find out the symbol you're looking for.

        timezone: Alternatively, a valid time zone for the returned timestamp, e.g., `US/Eastern`.
        
    """
    url = f"https://quotient.p.rapidapi.com/indexes/live"
    querystring = {'symbol': symbol, }
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def equity_financial(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return financial data (revenue, earnings, ratios, etc) for a given security."
    symbol: The symbol of the asset to look for, e.g., `AAPL` (Apple Inc), `F` (Ford Motor Company).
        
    """
    url = f"https://quotient.p.rapidapi.com/equity/financial"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def equity_live(symbol: str, timezone: str='UTC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return current market price data given the input parameters."
    symbol: The symbol of the asset to look for, e.g., `AAPL` (Apple Inc), `TSLA` (Tesla Inc).
        timezone: Alternatively, a valid time zone for the returned timestamp, e.g., `US/Eastern`.
        
    """
    url = f"https://quotient.p.rapidapi.com/equity/live"
    querystring = {'symbol': symbol, }
    if timezone:
        querystring['timezone'] = timezone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def options_prices(type: str, symbol: str, min_strike: int=50, min_expiry: str='2021-05-21', max_expiry: str='2023-12-14', max_strike: int=90, strike: int=122, expiration: str='2023-12-14', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return current options data given the input parameters."
    type: The type of option contract. (`Call` or `Put`).
        symbol: The symbol of the asset to look for, e.g., `AAPL` (Apple Inc), `MSFT` (Microsoft).
        min_strike: Alternatively, a lower bound for strike price, e.g., `50`. If given, strike range **[`min_strike`, `max_strike`]** takes priority over `strike` parameter.
        min_expiry: Alternatively, a lower bound for the expiration date (supported format is : **YYYY-mm-dd**), e.g., `2021-05-21`. If given, expiry range **[`min_expiry`, `max_expiry`]** takes priority over `expiration` parameter.
        max_expiry: Alternatively, an upper bound for the expiration date (supported format is : **YYYY-mm-dd**), e.g., `2021-12-14`. If given, expiry range **[`min_expiry`, `max_expiry`]** takes priority over `expiration` parameter.
        max_strike: Alternatively, an upper bound for strike price, e.g., `90`. If given, strike range **[`min_strike`, `max_strike`]** takes priority over `strike` parameter.
        strike: The strike price of the option, e.g., `122.00`.
        expiration: The expiration date of the option (supported format is : **YYYY-mm-dd**), e.g., `2021-12-14`.
        
    """
    url = f"https://quotient.p.rapidapi.com/options/prices"
    querystring = {'type': type, 'symbol': symbol, }
    if min_strike:
        querystring['min_strike'] = min_strike
    if min_expiry:
        querystring['min_expiry'] = min_expiry
    if max_expiry:
        querystring['max_expiry'] = max_expiry
    if max_strike:
        querystring['max_strike'] = max_strike
    if strike:
        querystring['strike'] = strike
    if expiration:
        querystring['expiration'] = expiration
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def equity_daily(is_from: str, to: str, symbol: str, adjust: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return end of day (daily) time series (Date, Open, High, Low, Close, Volume) given the input parameters."
    from: The query start date (supported format is : **YYYY-mm-dd**), e.g., `2018-04-01`.
        to: The query end date (supported format is : **YYYY-mm-dd**), e.g., `2020-04-21`.
        symbol: The symbol of the asset to look for, e.g., `AAPL` (Apple Inc), `MSFT` (Microsoft).
        adjust: Tell if price need to be adjusted.
        
    """
    url = f"https://quotient.p.rapidapi.com/equity/daily"
    querystring = {'from': is_from, 'to': to, 'symbol': symbol, }
    if adjust:
        querystring['adjust'] = adjust
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def equity_intraday(interval: int, to: str, is_from: str, symbol: str, adjust: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return intraday time series (Date, Open, High, Low, Close, Volume) given the input parameters."
    interval: 1-minute level time interval, e.g., `1` (1 min), `5` (5 min).
        to: The query end date (supported formats are : **YYYY-mm-dd HH:MM**, **YYYY-mm-dd**), e.g., `2020-04-21 10:30` or simply `2020-04-22`.
        from: The query start date (supported formats are : **YYYY-mm-dd HH:MM**, **YYYY-mm-dd**), e.g., `2020-04-21 10:00` or simply `2020-04-21`.
        symbol: The symbol of the asset to look for, e.g., `AAPL` (Apple Inc), `MSFT` (Microsoft).
        adjust: Tell if price need to be adjusted.
        
    """
    url = f"https://quotient.p.rapidapi.com/equity/intraday"
    querystring = {'interval': interval, 'to': to, 'from': is_from, 'symbol': symbol, }
    if adjust:
        querystring['adjust'] = adjust
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def indexes_intraday(interval: int, is_from: str, to: str, symbol: str, adjust: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return intraday time series (Date, Open, High, Low, Close, Volume) given the input parameters."
    interval: 1-minute level time interval, e.g., `1` (1 min), `5` (5 min).
        from: The query start date (supported formats are : **YYYY-mm-dd HH:MM**, **YYYY-mm-dd**), e.g., `2020-04-21 10:00` or simply `2020-04-21`.
        to: The query end date (supported formats are : **YYYY-mm-dd HH:MM**, **YYYY-mm-dd**), e.g., `2020-04-21 10:30` or simply `2020-04-22`.
        symbol: The symbol of the index to look for, e.g., `^GSPC` (S&P 500 index) or with suffix notation `SPX:INDEX` (S&P 500 index), `^GSPTSE` (TSX Composite Index) or with suffix notation `TXCX:INDEXCA` (TSX Composite Index). Valid suffixes are :

  - `:INDEX`: for world indices, e.g., `SPX:INDEX` (S&P 500 index) or `^GSPC`
  - `:INDEXUS`: for us specific indices, e.g., `SREN:INDEXUS` (S&P 500 Energy (Sector)) or `^GSPE`
  - `:INDEXCA`: for canadian specific indices, e.g., `TXCX:INDEXCA` (TSX Composite Index) or `^GSPTSE`
  - `:INDEXAU`: for australian specific indices, e.g., `XTO:INDEXAU` (ASX 100 Index) or `^ATOI`
  - `:INDEXEU`: for european specific indices, e.g., `BEL2I:INDEXEU` (BEL 20 Gr) or `BEL2I.BR`

  Please use the lookup endpoint to find out the symbol you're looking for.

        adjust: Tell if price need to be adjusted.
        
    """
    url = f"https://quotient.p.rapidapi.com/indexes/intraday"
    querystring = {'interval': interval, 'from': is_from, 'to': to, 'symbol': symbol, }
    if adjust:
        querystring['adjust'] = adjust
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def equity_splits(symbol: str, is_from: str, to: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return splits history data for a given security."
    symbol: The symbol of the asset to look for, e.g., `AAPL` (Apple Inc), `MSFT` (Microsoft Corporation).
        from: The query start date (supported format is : **YYYY-mm-dd**), e.g., `2019-01-01`.
        to: The query end date (supported format is : **YYYY-mm-dd**), e.g., `2020-04-17`.
        
    """
    url = f"https://quotient.p.rapidapi.com/equity/splits"
    querystring = {'symbol': symbol, 'from': is_from, 'to': to, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forex_signal(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a trading signal based on market sentiment, reliable indicators, analyst ratings and news. Signal : `1`=**buy**, `0`=**hold** and `-1`=**sell**. Confidence interval : [0%-100%]."
    symbol: The symbol of the currency pair to look for, e.g., `EUR/USD` or `EUR-USD` or `EURUSD`.
        
    """
    url = f"https://quotient.p.rapidapi.com/forex/signal"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def equity_earnings(to: str, is_from: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return earnings (EPS, quarterly) history data for a given security."
    to: The query end date (supported format is : **YYYY-mm-dd**), e.g., `2020-04-17`.
        from: The query start date (supported format is : **YYYY-mm-dd**), e.g., `2019-01-01`.
        symbol: The symbol of the asset to look for, e.g., `AAPL` (Apple Inc), `T` (AT&T Inc).
        
    """
    url = f"https://quotient.p.rapidapi.com/equity/earnings"
    querystring = {'to': to, 'from': is_from, 'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def forex_intraday(interval: int, symbol: str, to: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return intraday time series (Date, Open, High, Low, Close, Volume) given the input parameters."
    interval: 1-minute level time interval, e.g., `1` (1 min), `5` (5 min).
        symbol: The symbol of the currency pair to look for, e.g., `EUR/USD` or `EUR-USD` or `EURUSD`.
        to: The query end date (supported formats are : **YYYY-mm-dd HH:MM**, **YYYY-mm-dd**), e.g., `2020-04-21 10:30` or simply `2020-04-22`.
        from: The query start date (supported formats are : **YYYY-mm-dd HH:MM**, **YYYY-mm-dd**), e.g., `2020-04-21 10:00` or simply `2020-04-21`.
        
    """
    url = f"https://quotient.p.rapidapi.com/forex/intraday"
    querystring = {'interval': interval, 'symbol': symbol, 'to': to, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def options_historical(expiration: str, type: str, strike: int, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return historical options data given the input parameters."
    expiration: The expiration date of the option (supported format is : **YYYY-mm-dd**), e.g., `2018-04-20`.
        type: The type of option contract. (`Call` or `Put`).
        strike: The strike price of the option, e.g., `100`.
        symbol: The symbol of the asset to look for, e.g., `AAPL` (Apple Inc), `MSFT` (Microsoft).
        
    """
    url = f"https://quotient.p.rapidapi.com/options/historical"
    querystring = {'expiration': expiration, 'type': type, 'strike': strike, 'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def equity_signal(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return a trading signal based on market sentiment, reliable indicators, analyst ratings and news. Signal : `1`=**buy**, `0`=**hold** and `-1`=**sell**. Confidence interval : [0%-100%]."
    symbol: The symbol of the asset to look for, e.g., `AAPL` (Apple Inc), `F` (Ford Motor Company).
        
    """
    url = f"https://quotient.p.rapidapi.com/equity/signal"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "quotient.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

