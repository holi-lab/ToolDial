import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def trendings(date: str, region_code: str, hl: str='en-US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint used to display some of the trending search keywords on Google in a specific region and on a specific date."
    date: To display trend data for a specific date
        region_code: The region_code parameter is used to display data only for the specified region.
Example: **GB**, **ID**, **US**, etc.
To view the list of supported regions, please check the **/regions** endpoint.
        
    """
    url = f"https://google-trends8.p.rapidapi.com/trendings"
    querystring = {'date': date, 'region_code': region_code, }
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-trends8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def regions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint to retrieve a list of supported regions"
    
    """
    url = f"https://google-trends8.p.rapidapi.com/regions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-trends8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

