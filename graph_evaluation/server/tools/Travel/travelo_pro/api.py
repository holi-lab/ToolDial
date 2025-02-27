import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def travel_technology(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Travelopro provides travel API integration, third party API integration, xml API integration services for hotels, flights, car rentals, holiday packages and many more."
    
    """
    url = f"https://travelo-pro.p.rapidapi.comhttps://www.travelopro.com/api.php"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travelo-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

