import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_daily_information_for_all_companies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Tadawul Daily information for all companies"
    
    """
    url = f"https://saudi-exchange-stocks-tadawul.p.rapidapi.com/v1/getDailyInformation"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "saudi-exchange-stocks-tadawul.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_dividend(companyid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can use this API to get the whole dividend recorded in tadawul"
    
    """
    url = f"https://saudi-exchange-stocks-tadawul.p.rapidapi.com/v1/stock/getDividend"
    querystring = {'companyId': companyid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "saudi-exchange-stocks-tadawul.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stock_prices(period: str, companyid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can use this api to get stock prices using time period."
    period: You can use the following :
1D, 5D, 3M, 6M, 1Y, 5Y, AY
        
    """
    url = f"https://saudi-exchange-stocks-tadawul.p.rapidapi.com/v1/stock/getPrice"
    querystring = {'period': period, 'companyId': companyid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "saudi-exchange-stocks-tadawul.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

