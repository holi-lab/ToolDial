import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(to: str, is_from: str, depart_date: str, return_date: str='2015-04-07', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search"
    to: 3 letter airport code
        from: 3 letter airport code
        depart_date: yyyy-mm-dd
        return_date: yyyy-mm-dd
        
    """
    url = f"https://siddiq-such-flight-v1.p.rapidapi.com/search"
    querystring = {'to': to, 'from': is_from, 'depart-date': depart_date, }
    if return_date:
        querystring['return-date'] = return_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "siddiq-such-flight-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

