import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def consumer_address_search(limit: str, packages: str, query: str='123 Main St', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find relevant People in the Data Axle database"
    
    """
    url = f"https://data-axle-consumer-address-search.p.rapidapi.com/v1/people/search"
    querystring = {'limit': limit, 'packages': packages, }
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "data-axle-consumer-address-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

