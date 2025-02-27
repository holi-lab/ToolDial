import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def find_by_address(query: str='Farringdon Road', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "“Find By Address” API gets addresses for the given search term. This API provides a JSON interface to search UK addresses. This API gives access to Royal Mail PAF data and Geocoding."
    
    """
    url = f"https://find-by-address.p.rapidapi.com/v1/addresses"
    querystring = {}
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "find-by-address.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

