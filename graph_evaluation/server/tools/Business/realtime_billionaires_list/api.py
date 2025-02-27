import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def optional_get_all_billionaires_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Optional endpoint will  also return back  all the Billionaires  based on their **Rankings in Real-Time.**"
    
    """
    url = f"https://realtime-billionaires-list.p.rapidapi.com/rich"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-billionaires-list.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_billionaire_by_rank_using_query(rank: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint will return back the billionaire on that  Specific Rank
		**Note :** *The Rank Starts from*** 1**.
		**The Endpoint Example :** /rich/rank=1"
    
    """
    url = f"https://realtime-billionaires-list.p.rapidapi.com/rich"
    querystring = {'rank': rank, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-billionaires-list.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_billionaires_within_specific_range(end: str, start: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will return the Billionaires Data within the provided range.
		
		**Note** : *The Range starts from* **1**
		**Example:**  rich?start=1&end=10"
    
    """
    url = f"https://realtime-billionaires-list.p.rapidapi.com/rich"
    querystring = {'end': end, 'start': start, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-billionaires-list.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_billionaire_by_rank(rank: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Endpoint will return back the billionaire on that  Specific Rank
		**Note :** *The Rank Starts from*** 1**.
		**The Endpoint Example :** /rich/1"
    
    """
    url = f"https://realtime-billionaires-list.p.rapidapi.com/rich/{rank}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-billionaires-list.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_only_required_number_of_billionaires_data(limit: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will get you the Billionaires Data  within the limit
		
		**Note**: *The Limit Starts from* 1.
		**Endpoint Example** : /rich?limit=10"
    
    """
    url = f"https://realtime-billionaires-list.p.rapidapi.com/rich"
    querystring = {'limit': limit, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-billionaires-list.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_billionaires_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint will **return back all the Billionaires** based on their **Rankings in Real-Time.**"
    
    """
    url = f"https://realtime-billionaires-list.p.rapidapi.com/billionaires"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realtime-billionaires-list.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

