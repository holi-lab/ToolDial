import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_ndc_directory(limit: str='1', generic_name: str='levothyroxine', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search_NDC_directory"
    
    """
    url = f"https://open-fda.p.rapidapi.com/drug/ndc_directory/search"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if generic_name:
        querystring['generic_name'] = generic_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-fda.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

