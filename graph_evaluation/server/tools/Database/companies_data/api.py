import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def referential(page_number: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns list of companies"
    page_number: Pagination page number
        country: Two letter country ISO code (e.g. 'fr' for France)
        
    """
    url = f"https://companies-data1.p.rapidapi.com/api/v1/{country}/companies/{page_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def data(siren: str, country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns company details for one or more SIREN numbers"
    siren: Unique ID for each company
        country: Two letter country ISO code (e.g. 'fr' for France)
        
    """
    url = f"https://companies-data1.p.rapidapi.com/api/v1/{country}/profiles/{siren}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "companies-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

