import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_details_info_of_the_location_by_ip_address(ip: str, api_key: str='bee074b4c947a3d83e56d0c3803e50b06a66765611caef18cf180db6eb1580c6', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this API retrieve the location details of an IP address like region, country, city, zip code and etc."
    
    """
    url = f"https://ip2location8.p.rapidapi.com/get-info/{ip}"
    querystring = {}
    if api_key:
        querystring['api_key'] = api_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ip2location8.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

