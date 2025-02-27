import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_list_of_registered_symbols(symbolstype: int=3, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In This function you can find all the symbols that we are currentling streaming the latest price.
		
		Please keep calling this function and get your system updated  we adding new list of symbols everyday."
    
    """
    url = f"https://stock-market14.p.rapidapi.com/MarketData/GetListOfSymbolNames"
    querystring = {}
    if symbolstype:
        querystring['symbolsType'] = symbolstype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market14.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stock_prices(symboles: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In this API you can retrieve a list of stocks by comma seperator
		
		Ex.
		AMD,GOOG,TQQQ"
    
    """
    url = f"https://stock-market14.p.rapidapi.com/MarketData/GetLastPrices"
    querystring = {'symboles': symboles, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market14.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stock_price(symbole: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get single stock price"
    
    """
    url = f"https://stock-market14.p.rapidapi.com/MarketData/GetLastPrice"
    querystring = {'symbole': symbole, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stock-market14.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

