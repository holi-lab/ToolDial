import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_stochastics_cross_over_above_30(period2: str, interval: str, symbol: str, period1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API for stocks to check stochastics cross over above 30"
    period2: To date
        interval: Interval
        symbol: Stock
        period1: From date. Provide 100 bars between period 1 and period  2
        
    """
    url = f"https://stock-series-indicator-strategies.p.rapidapi.com/IndicatorStrategies/GetStochasticsCrossOverAbove30"
    querystring = {'period2': period2, 'interval': interval, 'symbol': symbol, 'period1': period1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-series-indicator-strategies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stochastics_cross_over_above_20(period1: str, symbol: str, interval: str, period2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API for stocks to check stochastics cross over above 20"
    period1: From date. Provide 100 bars between period 1 and period 2
        symbol: Stock
        interval: Interval
        period2: To date
        
    """
    url = f"https://stock-series-indicator-strategies.p.rapidapi.com/IndicatorStrategies/GetStochasticsCrossOverAbove20"
    querystring = {'period1': period1, 'symbol': symbol, 'interval': interval, 'period2': period2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-series-indicator-strategies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stochastics_cross_over_above_50(interval: str, period2: str, period1: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API for stocks to check stochastics cross over above 50"
    interval: Interval
        period2: To date
        period1: From date. Provide 100 bars between period 1 and period  2
        symbol: Stock
        
    """
    url = f"https://stock-series-indicator-strategies.p.rapidapi.com/IndicatorStrategies/GetStochasticsCrossOverAbove50"
    querystring = {'interval': interval, 'period2': period2, 'period1': period1, 'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-series-indicator-strategies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_historical_data(interval: str, period2: str, period1: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get historical stock data for the given period and time interval"
    interval: Interval
        period2: To date
        period1: From date. Provide 100 bars between period1 and period2
        symbol: Stock
        
    """
    url = f"https://stock-series-indicator-strategies.p.rapidapi.com/IndicatorStrategies/GetHistoricalData"
    querystring = {'interval': interval, 'period2': period2, 'period1': period1, 'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-series-indicator-strategies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rsi_7_period_above_30(interval: str, period2: str, symbol: str, period1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API for stocks to check RSI 7 period above 30"
    interval: Interval
        period2: To date
        symbol: Stock
        period1: From date. Provide 100 bars between period 1 and period 2
        
    """
    url = f"https://stock-series-indicator-strategies.p.rapidapi.com/IndicatorStrategies/GetRsi7PrdAbove30"
    querystring = {'interval': interval, 'period2': period2, 'symbol': symbol, 'period1': period1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-series-indicator-strategies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rsi_7_period_above_50(interval: str, period2: str, symbol: str, period1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API for stocks to check RSI 7 period above 50"
    interval: Interval
        period2: To date
        symbol: Stock
        period1: From date. Provide 100 bars between period 1 and period 2
        
    """
    url = f"https://stock-series-indicator-strategies.p.rapidapi.com/IndicatorStrategies/GetRsi7PrdAbove50"
    querystring = {'interval': interval, 'period2': period2, 'symbol': symbol, 'period1': period1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-series-indicator-strategies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rsi_14_period_above_50(interval: str, period1: str, symbol: str, period2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API for stocks to check RSI 14 period above 50"
    interval: Interval
        period1: Start date. Provide 100 bars between period 1 and period 2
        symbol: Stock
        period2: To date
        
    """
    url = f"https://stock-series-indicator-strategies.p.rapidapi.com/IndicatorStrategies/GetRsi14PrdAbove50"
    querystring = {'interval': interval, 'period1': period1, 'symbol': symbol, 'period2': period2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-series-indicator-strategies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rsi_2_below_10_and_above_90(period1: str, interval: str, symbol: str, period2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API for stocks to check RSI 2 below 10 and above 90"
    period1: From date. Provide 100 bars between period 1 and period 2
        interval: Interval
        symbol: Stock
        period2: To date
        
    """
    url = f"https://stock-series-indicator-strategies.p.rapidapi.com/IndicatorStrategies/GetRsi2Below10Above90"
    querystring = {'period1': period1, 'interval': interval, 'symbol': symbol, 'period2': period2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-series-indicator-strategies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rsi_14_period_above_30(period1: str, interval: str, symbol: str, period2: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API for stocks to check RSI 14 period above 30"
    period1: From date. Provide 100 bars between period 1 and period 2
        interval: Interval
        symbol: Stock
        period2: To date
        
    """
    url = f"https://stock-series-indicator-strategies.p.rapidapi.com/IndicatorStrategies/GetRsi14PrdAbove30"
    querystring = {'period1': period1, 'interval': interval, 'symbol': symbol, 'period2': period2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-series-indicator-strategies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rsi_14_period_above_20(period2: str, symbol: str, period1: str, interval: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API for stocks to check RSI 14 period above 20"
    period2: To date
        symbol: Stock
        period1: From date. Provide 100 bars between period 1 and period 2
        interval: Interval
        
    """
    url = f"https://stock-series-indicator-strategies.p.rapidapi.com/IndicatorStrategies/GetRsi14PrdAbove20"
    querystring = {'period2': period2, 'symbol': symbol, 'period1': period1, 'interval': interval, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-series-indicator-strategies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rsi_7_period_above_20(interval: str, period2: str, period1: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API for stocks to check RSI 7 period above 20"
    interval: Interval
        period2: To date
        period1: From date. Provide 100 bars between period 1 and period 2
        symbol: Stock
        
    """
    url = f"https://stock-series-indicator-strategies.p.rapidapi.com/IndicatorStrategies/GetRsi7PrdAbove20"
    querystring = {'interval': interval, 'period2': period2, 'period1': period1, 'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-series-indicator-strategies.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

