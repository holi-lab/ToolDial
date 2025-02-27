import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def essentials_lookup(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a list of all the tables used as a filter_name with the Essentials Lookup endpoints. The tables and their contents can be used to generate dropdowns and filters in your system when using the Essentials API."
    
    """
    url = f"https://essentials.p.rapidapi.com/lookup"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "essentials.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def essentials_lookup_filter_name_key_or_value(filter_name: str, key_or_value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lookup details on a specific filter using the known filter_name and key_or_value.  The resulting search_value can be passed as a search value in the Essentials API."
    filter_name: A list of all filters that can be used in Essentials endpoints. Found in the response of /essentials/lookup
        key_or_value: key of filterable items found in the response of /essentials/lookup/{filter_name}
        
    """
    url = f"https://essentials.p.rapidapi.com/lookup/{filter_name}/{key_or_value}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "essentials.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def essentials_lookup_filter_name(filter_name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides a list of all the human-readable search values within a specific filter_name table that can be used in search queries with the Essentials endpoints. Each array contains a table key, name value, and search value. The resulting search_value (s) can be passed as search values in the Essentials API."
    
    """
    url = f"https://essentials.p.rapidapi.com/lookup/{filter_name}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "essentials.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

