import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tripfro_travel_api(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "TripFro provides API integration services for travel agents, tour operators, and travel management companies. With JSON/XML API integration, travel companies can enhance their customers booking towards flight booking, hotel booking, car rental, holiday bookings, and many more as per the scope of API."
    
    """
    url = f"https://tripfro-travel.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tripfro-travel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

