import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def airbnbmarketmaker(bedrooms: int, country_code: str, coordinate: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It fetches up to 100 similar listings (similar compared to the listing detail given by you) and gets their next 365 days available / booked daily rates."
    
    """
    url = f"https://airbnb-market-maker.p.rapidapi.com/getBookingChanges"
    querystring = {'bedrooms': bedrooms, 'country_code': country_code, 'coordinate': coordinate, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-market-maker.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

