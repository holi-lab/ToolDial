import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(query: str, max: int=1, type: str='city', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retreive french cities/departments/regions base on the "query" parameter.
		
		By default, the first result you get is the perfect match."
    query: Examples values:
- Paris
- Bretagne
- Rennes

It's can be a city name, department name or region name.
        max: Specify max result, 1 to 10.

By default, the value is **10**.
        type: Filter results by type.

type values accepted :
- all
- city
- region
- department

By default, the value is **all**.
        
    """
    url = f"https://frenchcityregion.p.rapidapi.com/search"
    querystring = {'query': query, }
    if max:
        querystring['max'] = max
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "frenchcityregion.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

