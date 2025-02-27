import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_stock_forecast_api(symbol_name_exchange_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Forecastica stock forecasts API utilizes AI and our proprietary machine learning and neural network algorithm to generate 30 day stock forecasts for roughly 200,000 stock symbols that belong to 92 of the most well-known exchanges in the world that span the United States (around 15,000 symbols in total for NYSE, NASDAQ), Europe and Asia.  Each forecast includes open, close, low and high stock prices.   In addition, each forecast include a fap value which is the average of the open, close, low and high price and the fapPct which is a percentage value that represents a predicted percentage change in the predicted stock price from the day before to the next day.   In addition, each forecasting day includes a buy, hold and sell trading signal that is based on our proprietary machine learning trading algorithm.  Our trading signal is based on mathematically predicted turning points (relative minimum or maximum value of stock quotes vs. time)
		
		Our proprietary model is based on machine learning and neural networks. Simply put, our stock forecasts are driven by pattern recognition that enables the model to identify stock-purchasing trends. It then provides an objective recommendation on what action to take with any given stock!
		
		Our software will help you to hear the pulse of the stock market and get recommendations for the best BUY/SELL/HOLD moments!
		
		If you get zero results for the given stock symbol use the /exchanges/{EXCHANGE-SYMBOL}/stocks API to lookup the symbol for the given exchange because the symbol may have changed over time as a result you will need to use the most up to date symbol for the company."
    symbol_name_exchange_symbol: Consists of two parts: {SYMBOLNAME}-{EXCHANGE_SYMBOL}, then you can use, for example, MSFT.O-NASDAQ 
        
    """
    url = f"https://forecastica.p.rapidapi.com/api/forecasts/{symbol_name_exchange_symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecastica.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_historical_stock_prices_api(symbol_name_exchange_symbol: str, x_rapidapi_key: str=None, x_rapidapi_host: str=None, fromdate: str='2008-01-01', todate: str='2023-01-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Forecastica historical stock price API provides 5 years of historical end of day data for roughly 200,000 stock symbols that belong to 92 of the most well-known exchanges in the world and that span the United States (around 15,000 symbols in total for NYSE, NASDAQ), Europe and Asia. Each end of day stock price includes the open, close, low and high stock prices.   
		
		Use the ticker code to get the data for any company. For example, MSFT.O-NASDAQ consists of two parts separated by a dash: {SYMBOL_NAME}-{EXCHANGE_SYMBOL}
		
		To and From date must be in the following format YYYY-MM-DD
		
		If you get zero results for the given stock symbol use the /exchanges/{EXCHANGE-SYMBOL}/stocks API to lookup the symbol for the given exchange because the symbol may have changed over time as a result you will need to use the most up to date symbol for the company."
    symbol_name_exchange_symbol: Consists of two parts: {SYMBOLNAME}-{EXCHANGE_SYMBOL}, then you can use, for example, MSFT.O-NASDAQ 
        
    """
    url = f"https://forecastica.p.rapidapi.com/api/eod/{symbol_name_exchange_symbol}"
    querystring = {}
    if x_rapidapi_key:
        querystring['X-RapidAPI-Key'] = x_rapidapi_key
    if x_rapidapi_host:
        querystring['X-RapidAPI-Host'] = x_rapidapi_host
    if fromdate:
        querystring['fromDate'] = fromdate
    if todate:
        querystring['toDate'] = todate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecastica.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stock_symbols_by_exchange_api(exchange_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Forecastica APIs cover roughly 200,000 stock symbols which include 92 of the most well-known exchanges in the world that span the United States (around 15,000 symbols in total for NYSE, NASDAQ), Europe and Asia. This endpoint will return a valid list of stock symbols for the given stock exchange that can be leveraged for stock forecast and historical stock prices endpoint."
    
    """
    url = f"https://forecastica.p.rapidapi.com/api/exchanges/{exchange_symbol}/stocks"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecastica.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_list_of_exchanges_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Forecastica APIs cover roughly 200,000 stock symbols which include 92 of the most well-known exchanges in the world spanning the United States (around 15,000 symbols in total for NYSE, NASDAQ), Europe and Asia.  This endpoint will return a list of exchanges and its corresponding market symbol code needed for stock symbol queries."
    
    """
    url = f"https://forecastica.p.rapidapi.com/api/exchanges"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forecastica.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

