import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def b2b_travel_portal_development_with_api_integration(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TravelPD offers affordable travel portal development solutions with travel API integration and travel portal software. "
    
    """
    url = f"https://travel-api-s.p.rapidapi.comb2b-travel-portal-development.p.rapidapi.com"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-api-s.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

