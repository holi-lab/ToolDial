import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_ownersearch(search_keyword: str='netflix', postcode: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Does a search on **owner name** or **postcode** and returns details about the trademarks found for that owner. Duplicate owner results can be returned by this endpoint, especially when an owner had an address change.
		
		You can use literal 'null' if you don't want to use a param. So you can either search just for a keyword or just for a postcode or together if you want to filter by both."
    search_keyword: This is the search keyword such as 'ferr' that will return results for 'ferrari'.
You can use literal 'null' if you don't want to use this param. 
        postcode: This is the postcode such as '95066' that will return results for owners with that postcode in their application. Be aware attorneys might omit postcode from fillings, so those results cannot be returned as they don't have postcode attached as metadata. 

You can use literal 'null' if you don't want to use this param. 
        
    """
    url = f"https://uspto-trademark.p.rapidapi.com/v1/ownerSearch/{search_keyword}/{postcode}"
    querystring = {}
    if search_keyword:
        querystring['search_keyword'] = search_keyword
    if postcode:
        querystring['postcode'] = postcode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uspto-trademark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_trademarkavailable(keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "returns if the given keyword is available"
    
    """
    url = f"https://uspto-trademark.p.rapidapi.com/v1/trademarkAvailable/{keyword}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uspto-trademark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_trademarksearch(keyword: str, searchtype: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Does a keyword search and returns trademark name, serial number,services code, status, owner, address, filing date and registration date."
    searchtype: Default searchType is \"active\", which don't include Dead trademarks, but if you set \"all\" you see also expired trademarks. 
        
    """
    url = f"https://uspto-trademark.p.rapidapi.com/v1/trademarkSearch/{keyword}/{searchtype}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uspto-trademark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_serialsearch(serial_number: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Does a serial number search and returns details about the trademark attributes."
    
    """
    url = f"https://uspto-trademark.p.rapidapi.com/v1/serialSearch/{serial_number}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uspto-trademark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v1_databasestatus(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns info about the freshness of the Trademark Search API database.
		
		`last_update_date` holds the date of the newest entry from USPTO filling entries that is synced to the API. 
		
		`latest_trademark` array holds 10 random samples from the most recent confirmed registered trademarks, the `keyword`, the `registration_number` and the `registration_date` . Please note the date could be a few days back as not every day are published new registration of trademarks. They publish daily new partial filings and abandoned trademarks."
    
    """
    url = f"https://uspto-trademark.p.rapidapi.com/v1/databaseStatus"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uspto-trademark.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

