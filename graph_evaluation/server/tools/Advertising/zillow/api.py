import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def propertyextendedsearch(x_rapidapi_host: str='zillow57.p.rapidapi.com', x_rapidapi_key: str='96135938aamshdb821aa8d6a9ceep16775fjsne837923e82b4', status_type: str='ForSale', page: str='1', location: str='IN', home_type: str='Houses', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "propertyExtendedSearch"
    
    """
    url = f"https://zillow57.p.rapidapi.com/propertyExtendedSearch"
    querystring = {}
    if x_rapidapi_host:
        querystring['X-RapidAPI-Host'] = x_rapidapi_host
    if x_rapidapi_key:
        querystring['X-RapidAPI-Key'] = x_rapidapi_key
    if status_type:
        querystring['status_type'] = status_type
    if page:
        querystring['page'] = page
    if location:
        querystring['location'] = location
    if home_type:
        querystring['home_type'] = home_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zillow57.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

