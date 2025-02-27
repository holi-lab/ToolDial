import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_stock_by_id_stocks_stock_id_get(stock_id: str, should_return: str='False', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://finvaley.p.rapidapi.com/stocks/{stock_id}"
    querystring = {}
    if should_return:
        querystring['should_return'] = should_return
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finvaley.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stock_preview_by_id_stocks_stock_id_preview_get(stock_id: str, should_return: str='False', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://finvaley.p.rapidapi.com/stocks/{stock_id}/preview"
    querystring = {}
    if should_return:
        querystring['should_return'] = should_return
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finvaley.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stock_detailed_by_id_stocks_stock_id_detailed_get(stock_id: str, should_return: str='False', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://finvaley.p.rapidapi.com/stocks/{stock_id}/detailed"
    querystring = {}
    if should_return:
        querystring['should_return'] = should_return
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finvaley.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

