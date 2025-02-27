import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def postcodes_postcode(postcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets addresses for the given UK PostCode"
    postcode: Pass a PostCode value for looking up addresses
        
    """
    url = f"https://uk-postcode-api.p.rapidapi.com/postcodes/{postcode}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-postcode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addresses_autocomplete(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Suggests address results for a given search term."
    query: pass a search term for address suggestions for a complete or partial address query
        
    """
    url = f"https://uk-postcode-api.p.rapidapi.com/addresses/autocomplete"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-postcode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addresses(query: str, limit: int=None, page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets addresses for given search term."
    query: Pass a search term to get addresses matching search query
        limit: Maximum number of records to return
        page: Starting index for the record
        
    """
    url = f"https://uk-postcode-api.p.rapidapi.com/addresses"
    querystring = {'query': query, }
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-postcode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def udprn_udprn(udprn: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets addresses for given search term."
    udprn: Pass a search term to get addresses matching search query
        
    """
    url = f"https://uk-postcode-api.p.rapidapi.com/udprn/{udprn}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uk-postcode-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

