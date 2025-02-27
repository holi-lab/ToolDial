import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def time_series_daily(function: str, symbol: str, datatype: str='json', outputsize: str='compact', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns daily time series (date, daily open, daily high, daily low, daily close, daily volume) of the global equity specified, covering 20+ years of historical data."
    function: The time series of your choice. 
        symbol: The name of the equity of your choice.
        datatype: Strings json and csv are accepted with the following specifications: json returns the daily time series in JSON format; csv returns the time series as a CSV (comma separated value) file.
        outputsize: Strings compact and full are accepted with the following specifications: compact returns only the latest 100 data points; full returns the full-length time series of 20+ years of historical data. The \\\"compact\\\" option is recommended if you would like to reduce the data size of each API call.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'function': function, 'symbol': symbol, }
    if datatype:
        querystring['datatype'] = datatype
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_series_weekly(function: str, symbol: str='MSFT', datatype: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns weekly time series (last trading day of each week, weekly open, weekly high, weekly low, weekly close, weekly volume) of the global equity specified, covering 20+ years of historical data."
    function: The time series of your choice. 
        symbol: The name of the equity of your choice. 
        datatype: By default, datatype=json. Strings json and csv are accepted with the following specifications: json returns the intraday time series in JSON format; csv returns the time series as a CSV (comma separated value) file.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'function': function, }
    if symbol:
        querystring['symbol'] = symbol
    if datatype:
        querystring['datatype'] = datatype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_series_monthly_adjusted(symbol: str, function: str, datatype: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns monthly adjusted time series (last trading day of each month, monthly open, monthly high, monthly low, monthly close, monthly adjusted close, monthly volume, monthly dividend) of the equity specified, covering 20+ years of historical data."
    symbol: The equity of your choice.
        function: The time series of your choice.
        datatype: Strings json and csv are accepted with the following specifications: json returns the monthly time series in JSON format; csv returns the time series as a CSV (comma separated value) file.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'symbol': symbol, 'function': function, }
    if datatype:
        querystring['datatype'] = datatype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_series_daily_adjusted(function: str, symbol: str, datatype: str='json', outputsize: str='compact', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns daily time series (date, daily open, daily high, daily low, daily close, daily volume, daily adjusted close, and split/dividend events) of the global equity specified, covering 20+ years of historical data."
    function: The time series of your choice.
        symbol: The name of the equity of your choice. 
        datatype: By default, datatype=json. Strings json and csv are accepted with the following specifications: json returns the intraday time series in JSON format; csv returns the time series as a CSV (comma separated value) file.
        outputsize: Strings compact and full are accepted with the following specifications: compact returns only the latest 100 data points; full returns the full-length time series of 20+ years of historical data. The "compact" option is recommended if you would like to reduce the data size of each API call.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'function': function, 'symbol': symbol, }
    if datatype:
        querystring['datatype'] = datatype
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_endpoint(keywords: str, function: str, datatype: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Looking for some specific symbols or companies? We've got you covered! The Search Endpoint returns the best-matching symbols and market information based on keywords of your choice. The search results also contain match scores that provide you with the full flexibility to develop your own search and filtering logic."
    keywords: A text string of your choice. 
        function: The API function of your choice.
        datatype: Strings json and csv are accepted with the following specifications: json returns the search results in JSON format; csv returns the search results as a CSV (comma separated value) file.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'keywords': keywords, 'function': function, }
    if datatype:
        querystring['datatype'] = datatype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_series_weekly_adjusted(function: str, symbol: str, datatype: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns weekly adjusted time series (last trading day of each week, weekly open, weekly high, weekly low, weekly close, weekly adjusted close, weekly volume, weekly dividend) of the global equity specified, covering 20+ years of historical data."
    function: The time series of your choice. 
        symbol: The equity of your choice.
        datatype: By default, datatype=json. Strings json and csv are accepted with the following specifications: json returns the intraday time series in JSON format; csv returns the time series as a CSV (comma separated value) file.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'function': function, 'symbol': symbol, }
    if datatype:
        querystring['datatype'] = datatype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quote_endpoint(function: str, symbol: str, datatype: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A lightweight alternative to the time series APIs, this service returns the price and volume information for a security of your choice."
    function: The API function of your choice.
        symbol: The equity of your choice.
        datatype: By default, datatype=json. Strings json and csv are accepted with the following specifications: json returns the intraday time series in JSON format; csv returns the time series as a CSV (comma separated value) file.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'function': function, 'symbol': symbol, }
    if datatype:
        querystring['datatype'] = datatype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_series_intraday(interval: str, function: str, symbol: str, datatype: str='json', output_size: str='compact', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns intraday time series (timestamp, open, high, low, close, volume) of the equity specified."
    interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min
        function: The time series of your choice.
        symbol: The equity of your choice.
        datatype: By default, datatype=json. Strings json and csv are accepted with the following specifications: json returns the intraday time series in JSON format; csv returns the time series as a CSV (comma separated value) file.
        output_size: Strings compact and full are accepted with the following specifications: compact returns only the latest 100 data points in the intraday time series; full returns the full-length intraday time series. The \"compact\" option is recommended if you would like to reduce the data size of each API call.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'interval': interval, 'function': function, 'symbol': symbol, }
    if datatype:
        querystring['datatype'] = datatype
    if output_size:
        querystring['output_size'] = output_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def time_series_monthly(symbol: str, function: str, datatype: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns monthly time series (last trading day of each month, monthly open, monthly high, monthly low, monthly close, monthly volume) of the global equity specified, covering 20+ years of historical data."
    symbol: The equity of your choice.
        function: The time series of your choice.
        datatype: By default, datatype=json. Strings json and csv are accepted with the following specifications: json returns the intraday time series in JSON format; csv returns the time series as a CSV (comma separated value) file.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'symbol': symbol, 'function': function, }
    if datatype:
        querystring['datatype'] = datatype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def digital_currency_weekly(market: str, function: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the weekly historical time series for a digital currency (e.g., BTC) traded on a specific market (e.g., CNY/Chinese Yuan), refreshed daily at midnight (UTC). Prices and volumes are quoted in both the market-specific currency and USD."
    market: The exchange market of your choice. It can be any of the market in the market list. For example: market=CNY.
        function: The time series of your choice.
        symbol: The digital/crypto currency of your choice. It can be any of the currencies in the digital currency list. For example: symbol=BTC.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'market': market, 'function': function, 'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def technical_indicators(time_period: int, interval: str, series_type: str, function: str, symbol: str, datatype: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For more information on how to build all the different technical indicators there are, [please visit here](https://www.alphavantage.co/documentation/#technical-indicators)."
    time_period: Number of data points used to calculate each moving average value. Positive integers are accepted (e.g., time_period=60, time_period=200)
        interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
        series_type: The desired price type in the time series. Four types are supported: close, open, high, low
        function: The technical indicator of your choice. In this case, function=SMA
        symbol: The name of the security of your choice. For example: symbol=MSFT
        datatype: By default, datatype=json. Strings json and csv are accepted with the following specifications: json returns the daily time series in JSON format; csv returns the time series as a CSV (comma separated value) file.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'time_period': time_period, 'interval': interval, 'series_type': series_type, 'function': function, 'symbol': symbol, }
    if datatype:
        querystring['datatype'] = datatype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_exchange_rate_crypto(from_currency: str, function: str, to_currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the realtime exchange rate for any pair of digital currency (e.g., Bitcoin) or physical currency (e.g., USD)."
    from_currency: The cryptocurrency you want to convert from 
        function: The function of your choice. (In this case CURRENCY_EXCHANGE_RATE)
        to_currency: The currency you want to convert to
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'from_currency': from_currency, 'function': function, 'to_currency': to_currency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fx_intraday(function: str, interval: str, to_symbol: str, from_symbol: str, datatype: str='json', outputsize: str='compact', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns intraday time series (timestamp, open, high, low, close) of the FX currency pair specified, updated realtime."
    function: The time series of your choice.
        interval: Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min
        to_symbol: A three-letter symbol from the forex currency list. For example: to_symbol=USD
        from_symbol: A three-letter symbol from the forex currency list. For example: from_symbol=EUR
        datatype: By default, datatype=json. Strings json and csv are accepted with the following specifications: json returns the intraday time series in JSON format; csv returns the time series as a CSV (comma separated value) file.
        outputsize: By default, outputsize=compact. Strings compact and full are accepted with the following specifications: compact returns only the latest 100 data points in the intraday time series; full returns the full-length intraday time series. The "compact" option is recommended if you would like to reduce the data size of each API call.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'function': function, 'interval': interval, 'to_symbol': to_symbol, 'from_symbol': from_symbol, }
    if datatype:
        querystring['datatype'] = datatype
    if outputsize:
        querystring['outputsize'] = outputsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fx_weekly(function: str, from_symbol: str, to_symbol: str, datatype: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the weekly time series (timestamp, open, high, low, close) of the FX currency pair specified, updated realtime.  The latest data point is the price information for the week (or partial week) containing the current trading day, updated realtime."
    function: The time series of your choice.
        from_symbol: A three-letter symbol from the forex currency list. For example: from_symbol=EUR
        to_symbol: A three-letter symbol from the forex currency list. For example: to_symbol=USD
        datatype: By default, datatype=json. Strings json and csv are accepted with the following specifications: json returns the weekly time series in JSON format; csv returns the time series as a CSV (comma separated value) file.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'function': function, 'from_symbol': from_symbol, 'to_symbol': to_symbol, }
    if datatype:
        querystring['datatype'] = datatype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def customizable_endpoint(function: str, symbol: str='TSLA', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Build your own query from the [documentation, located here.](https://www.alphavantage.co/documentation/) There is one master catch-all endpoint, but the additional following "endpoints" give you samples of how to build URLs."
    function: The function of you choice (This is required for all API calls)
        symbol: Any additional optional parameters may be added. Some functions have multiple required parameters, check the documentation for more information. 
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'function': function, }
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fx_daily(from_symbol: str, function: str, to_symbol: str='USD', outputsize: str='compact', datatype: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the daily time series (timestamp, open, high, low, close) of the FX currency pair specified, updated realtime."
    from_symbol: A three-letter symbol from the forex currency list. For example: from_symbol=EUR
        function: The time series of your choice.
        to_symbol: A three-letter symbol from the forex currency list. For example: to_symbol=USD
        outputsize: By default, outputsize=compact. Strings compact and full are accepted with the following specifications: compact returns only the latest 100 data points in the daily time series; full returns the full-length daily time series. The "compact" option is recommended if you would like to reduce the data size of each API call.
        datatype: By default, datatype=json. Strings json and csv are accepted with the following specifications: json returns the daily time series in JSON format; csv returns the time series as a CSV (comma separated value) file.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'from_symbol': from_symbol, 'function': function, }
    if to_symbol:
        querystring['to_symbol'] = to_symbol
    if outputsize:
        querystring['outputsize'] = outputsize
    if datatype:
        querystring['datatype'] = datatype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def digital_currency_daily(market: str, symbol: str, function: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the daily historical time series for a digital currency (e.g., BTC) traded on a specific market (e.g., CNY/Chinese Yuan), refreshed daily at midnight (UTC). Prices and volumes are quoted in both the market-specific currency and USD."
    market: The exchange market of your choice. It can be any of the market in the market list. For example: market=CNY.
        symbol: The digital/crypto currency of your choice. It can be any of the currencies in the digital currency list. For example: symbol=BTC.
        function: The time series of your choice.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'market': market, 'symbol': symbol, 'function': function, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_exchange_rate(to_currency: str, function: str, from_currency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the realtime exchange rate for any pair of digital currency (e.g., Bitcoin) and physical currency (e.g., USD). Data returned for physical currency (Forex) pairs also include realtime bid and ask prices."
    to_currency: The currency you want to convert to
        function: The function of your choice. (In this case  CURRENCY_EXCHANGE_RATE)
        from_currency: The currency you want to convert
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'to_currency': to_currency, 'function': function, 'from_currency': from_currency, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fx_monthly(from_symbol: str, to_symbol: str, function: str, datatype: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the monthly time series (timestamp, open, high, low, close) of the FX currency pair specified, updated realtime.  The latest data point is the prices information for the month (or partial month) containing the current trading day, updated realtime."
    from_symbol: A three-letter symbol from the forex currency list. For example: from_symbol=EUR
        to_symbol: A three-letter symbol from the forex currency list. For example: from_symbol=USD
        function: The time series of your choice.
        datatype: By default, datatype=json. Strings json and csv are accepted with the following specifications: json returns the monthly time series in JSON format; csv returns the time series as a CSV (comma separated value) file.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'from_symbol': from_symbol, 'to_symbol': to_symbol, 'function': function, }
    if datatype:
        querystring['datatype'] = datatype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def digital_currency_monthly(function: str, market: str, symbol: str='BTC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns the monthly historical time series for a digital currency (e.g., BTC) traded on a specific market (e.g., CNY/Chinese Yuan), refreshed daily at midnight (UTC). Prices and volumes are quoted in both the market-specific currency and USD."
    function: The time series of your choice.
        market: The exchange market of your choice. It can be any of the market in the market list. For example: market=CNY.
        symbol: The digital/crypto currency of your choice. It can be any of the currencies in the digital currency list. For example: symbol=BTC.
        
    """
    url = f"https://alpha-vantage.p.rapidapi.com/query"
    querystring = {'function': function, 'market': market, }
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

