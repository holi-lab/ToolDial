import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def airline_travel(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Flight Booking Engine comes with advanced and features of third-party integration, GDS connectivity, inventory management, quotation management, B2B/B2C booking engine, one-way, round-trip, multi-city search option, booking management, reporting, customer management, custom design and layout, multi-language, multi-currency, payment gateway integration and more."
    
    """
    url = f"https://airline-travel.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airline-travel.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

