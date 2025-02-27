import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def line_chart(country: str, indicator: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Line chart for given indicator and countries"
    country: Comma separated country codes. Maximum 5 countries are allowed.
        
    """
    url = f"https://word-bank-world-development-indicators.p.rapidapi.com/charts/line"
    querystring = {'country': country, 'indicator': indicator, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-bank-world-development-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def bar_chart(indicator: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Bar chart for given indicator and countries"
    country: Comma separated country codes. A maximum of 5 countries are allowed.
        
    """
    url = f"https://word-bank-world-development-indicators.p.rapidapi.com/charts/bar"
    querystring = {'indicator': indicator, 'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-bank-world-development-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def indicator_data(indicator: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get indicator data by country (region) code and indicator code. The indicator code can be found by querying indicators with keywords under `/indicators` endpoint."
    country: ISO 3166 alpha-3 country codes
        
    """
    url = f"https://word-bank-world-development-indicators.p.rapidapi.com/data"
    querystring = {'indicator': indicator, 'country': country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-bank-world-development-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def indicator_detail(indicatorcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get individual indicator detail information"
    
    """
    url = f"https://word-bank-world-development-indicators.p.rapidapi.com/indicators/{indicatorcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-bank-world-development-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def indicator_list(pagesize: int=20, page: int=1, q: str='GDP', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all indicators. The list is paginated and can be filtered with query parameters."
    
    """
    url = f"https://word-bank-world-development-indicators.p.rapidapi.com/indicators"
    querystring = {}
    if pagesize:
        querystring['pageSize'] = pagesize
    if page:
        querystring['page'] = page
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "word-bank-world-development-indicators.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

