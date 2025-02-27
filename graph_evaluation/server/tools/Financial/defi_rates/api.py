import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_rates_data_by_vendor(vendor: str, limit: int=100, skip: int=0, date_to: str='2022-01-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rates data by supported vendor, with any market or assets supported by API"
    date_to: Filter by date include inserted value.
        
    """
    url = f"https://defi-rates.p.rapidapi.com/rates/{vendor}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if skip:
        querystring['skip'] = skip
    if date_to:
        querystring['date_to'] = date_to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defi-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vendors(limit: int=100, skip: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get supported vendors list"
    
    """
    url = f"https://defi-rates.p.rapidapi.com/vendors"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if skip:
        querystring['skip'] = skip
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defi-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rates_data_by_vendor_and_market_and_asset(market: str, vendor: str, asset: str, limit: int=100, skip: int=0, date_to: str='2022-01-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rates data by supported vendor and market and certain asset supported by API"
    date_to: Filter by date include inserted value.
        
    """
    url = f"https://defi-rates.p.rapidapi.com/rates/{vendor}/{market}/{asset}"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if skip:
        querystring['skip'] = skip
    if date_to:
        querystring['date_to'] = date_to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defi-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rates_data_by_vendor_and_market(market: str, vendor: str, skip: int=0, date_to: str='2022-01-01', limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rates data by supported vendor and market with any assets supported by API"
    date_to: Filter by date include inserted value.
        
    """
    url = f"https://defi-rates.p.rapidapi.com/rates/{vendor}/{market}"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if date_to:
        querystring['date_to'] = date_to
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defi-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_rates_data(skip: int=0, limit: int=100, date_to: str='2022-01-01', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get rates data include all vendors, market and assets supported by API"
    date_to: Filter by date include inserted value.
        
    """
    url = f"https://defi-rates.p.rapidapi.com/rates"
    querystring = {}
    if skip:
        querystring['skip'] = skip
    if limit:
        querystring['limit'] = limit
    if date_to:
        querystring['date_to'] = date_to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "defi-rates.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

