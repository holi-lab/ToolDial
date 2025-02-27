import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def stockdividendinformation(stock_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns 4 summary properties of a stock related to dividends. 
		1. Stock latest dividend date.
		2. Stock dividend yield.
		3. Stock annual dividend.
		4. Stock dividend frequency."
    
    """
    url = f"https://stock-finance.p.rapidapi.com/stock/dividend/{stock_symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stockinformation(stock_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a general endpoint providing response of the general stock information."
    
    """
    url = f"https://stock-finance.p.rapidapi.com/stock/information/{stock_symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stockchartpattern(stock_symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint returns the prediction on how the stock chart pattern looks like. There could be three types of responses:-
		1. Bearish
		2. Bullish
		3. Neutral"
    
    """
    url = f"https://stock-finance.p.rapidapi.com/stock/chart_pattern/{stock_symbol}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

