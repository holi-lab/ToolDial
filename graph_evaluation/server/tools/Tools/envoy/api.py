import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_metadata_from_date(from_date: str, to_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the metadata associated with all entries on the account starting from `from_date` and ending on `to_date`. Note that the S3 URLs that are returned expire after one hour."
    from_date: Ruby `DateTime.parse` compatible string picking which day to start from
        to_date: Ruby `DateTime.parse` compatible string picking which day to end on
        
    """
    url = f"https://baskarm28-envoy-v1.p.rapidapi.com/entries.json"
    querystring = {'from_date': from_date, 'to_date': to_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-envoy-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def specific_metadata_retrieve(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the metadata associated with a specific entry_id. Note that the S3 URLs that are returned expire after one hour."
    
    """
    url = f"https://baskarm28-envoy-v1.p.rapidapi.com/entries/ENTRY_ID_HERE.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-envoy-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def retrieve_all(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the metadata associated with all entries on the account. Note that the S3 URLs that are returned expire after one hour"
    
    """
    url = f"https://baskarm28-envoy-v1.p.rapidapi.com/entries.json"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "baskarm28-envoy-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

