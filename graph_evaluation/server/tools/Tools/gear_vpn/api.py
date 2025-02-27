import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_vpn_configurations(limit: int=10, country: str='Japan', offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Fetch all/filtered VPN configurations. These response contains TCP & UDP configs which you can directly load into your VPN application.
		
		The API supports pagination & various filters through query parameters.
		
		Note: These configurations are refreshed every 4 hours, so you better not cache them locally otherwise you might not be able to connect them when expired."
    country: (Optional) Filter results by country which you can get from \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"List all available countries\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" endpoint.
        
    """
    url = f"https://gear-vpn.p.rapidapi.com/all"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if country:
        querystring['country'] = country
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gear-vpn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_all_available_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API lists all the available countries whose VPN configurations are available."
    
    """
    url = f"https://gear-vpn.p.rapidapi.com/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "gear-vpn.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

