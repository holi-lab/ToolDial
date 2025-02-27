import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def auto_complete(q: str, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Edamam provides a convenient autocomplete functionality which can be implemented for use when searching for ingredients. Just send in the current query as the "q" parameter and the number of suggestions you wish to receive as the "limit" parameter. 
		 
		 <b>Access Point:</b> https://api.edamam.com/auto-complete"
    q: Query text. For example q=chi. This or the r parameter are required
        limit: response limit
        
    """
    url = f"https://edamam-food-and-grocery-database.p.rapidapi.com/auto-complete"
    querystring = {'q': q, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edamam-food-and-grocery-database.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

