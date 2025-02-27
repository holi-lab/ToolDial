import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def pull_filing(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a specific filing"
    id: accession_id of the request ie "0000950123-20-009191"
        
    """
    url = f"https://sec-securities-and-exchange-dataset.p.rapidapi.com/pull_filing"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-securities-and-exchange-dataset.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def pull_13f_json(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a specific 13-F filing"
    id: accession_id of the request ie "0000950123-20-009191"
        
    """
    url = f"https://sec-securities-and-exchange-dataset.p.rapidapi.com/pull_13f_json"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-securities-and-exchange-dataset.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def query(companyname: str='Black', cik: str='19475', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Query functionality supports a number of filters. The result will only include 20 results, but has the ability
		to filter for specific results."
    companyname: company name
        cik: company identified
        
    """
    url = f"https://sec-securities-and-exchange-dataset.p.rapidapi.com/query"
    querystring = {}
    if companyname:
        querystring['companyname'] = companyname
    if cik:
        querystring['cik'] = cik
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-securities-and-exchange-dataset.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_entity_name(companyid: str='AAPL', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search functionality supports a number of filters. The result will only include 20 results, but has the ability
		to filter for specific results.
		"
    companyid: company identifier, cik, or ticker
        
    """
    url = f"https://sec-securities-and-exchange-dataset.p.rapidapi.com/search"
    querystring = {}
    if companyid:
        querystring['companyid'] = companyid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sec-securities-and-exchange-dataset.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

