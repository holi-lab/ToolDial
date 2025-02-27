import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(accept: str, content_type: str, pagenumber: int, pagesize: int, search: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using data made available by the USDA, perform a search on common foods and drinks to get as much macronutrient information as possible. Click here to get the USDA API key: https://fdc.nal.usda.gov/api-key-signup.html"
    
    """
    url = f"https://macronutrient-search.p.rapidapi.com/search"
    querystring = {'Accept': accept, 'Content-Type': content_type, 'pageNumber': pagenumber, 'pageSize': pagesize, 'search': search, 'apiKey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "macronutrient-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

