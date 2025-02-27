import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcurrentbyvaluegrowthmomentum(momentum: str, growth: str, value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of stocks according to current value, growth and momentum information."
    
    """
    url = f"https://earningsdata.p.rapidapi.com/rank/GetCurrentByValueGrowthMomentum/"
    querystring = {'momentum': momentum, 'growth': growth, 'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "earningsdata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcurrentbysymbol(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current value, growth and momentum information for a single stock."
    
    """
    url = f"https://earningsdata.p.rapidapi.com/rank/GetCurrent/"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "earningsdata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbymonth(month: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Earning Data By Month"
    
    """
    url = f"https://earningsdata.p.rapidapi.com/dates/GetByMonth"
    querystring = {'month': month, 'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "earningsdata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbydate(date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get earnings data by date."
    
    """
    url = f"https://earningsdata.p.rapidapi.com/dates/GetByDate/"
    querystring = {'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "earningsdata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbystock(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get earnings data by symbol."
    
    """
    url = f"https://earningsdata.p.rapidapi.com/dates/GetByStock/"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "earningsdata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

