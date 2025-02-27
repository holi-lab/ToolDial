import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def address_autosuggest(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""Address Autosuggest" API buit to suggests address results for a given search term. This API provides a JSON interface to extract address suggestions for a complete or partial address query.
		
		The Autocomplete API suggests partial address results for a given query."
    
    """
    url = f"https://address-autosuggest.p.rapidapi.com/v1/addresses/autocomplete"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "address-autosuggest.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

