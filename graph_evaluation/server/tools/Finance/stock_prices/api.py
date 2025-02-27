import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def max_historical_quarterly_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Max quarterly historically adjusted Open, High, Low, and Close prices.  Monthly Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/max-3mo-interval"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def max_historical_monthly_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Max monthly historically adjusted Open, High, Low, and Close prices.  Monthly Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/max-1mo-interval"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_day_historical_daily_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "1 Day historically adjusted Open, High, Low, and Close prices.  Daily Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/1d"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_2_year_historical_quarterly_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "2 Year quarterly historically adjusted Open, High, Low, and Close prices.  Quarterly Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/2y-3mo-interval"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_2_year_historical_monthly_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "2 Year monthly historically adjusted Open, High, Low, and Close prices.  Monthly Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/2y-1mo-interval"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_5_day_historical_daily_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "5 Day historically adjusted Open, High, Low, and Close prices.  Daily Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/5d"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_5_year_historical_quarterly_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "5 Year quarterly historically adjusted Open, High, Low, and Close prices.  Quarterly Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/5y-3mo-interval"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_5_year_historical_monthly_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "5 Year monthly historically adjusted Open, High, Low, and Close prices.  Monthly Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/5y-1mo-interval"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_month_historical_daily_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "1 Month historically adjusted Open, High, Low, and Close prices.  Daily Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/1mo"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ytd_historical_daily_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Year to Date historically adjusted Open, High, Low, and Close prices.  Daily Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/ytd"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_10_year_historical_monthly_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "10 Year monthly historically adjusted Open, High, Low, and Close prices.  Monthly Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/10y-1mo-interval"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_10_year_historical_quarterly_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "10 Year quarterly historically adjusted Open, High, Low, and Close prices.  Quarterly Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/10y-3mo-interval"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_3_month_historical_daily_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "3 Month historically adjusted Open, High, Low, and Close prices.  Daily Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/3mo"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_6_month_historical_daily_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "6 Month historically adjusted Open, High, Low, and Close prices.  Daily Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/6mo"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_1_year_historical_daily_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "1 Year historically adjusted Open, High, Low, and Close prices.  Daily Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/1y"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_5_year_historical_daily_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "5 Year historically adjusted Open, High, Low, and Close prices.  Daily Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/5y"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def max_historical_daily_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Max historically adjusted Open, High, Low, and Close prices.  Daily Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/max"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_2_year_historical_daily_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "2 Year historically adjusted Open, High, Low, and Close prices.  Daily Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/2y"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_10_year_historical_daily_prices(ticker: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "10 Year historically adjusted Open, High, Low, and Close prices.  Daily Volume, Dividend, and Split information is also included."
    
    """
    url = f"https://stock-prices2.p.rapidapi.com/api/v1/resources/stock-prices/10y"
    querystring = {'ticker': ticker, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-prices2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

