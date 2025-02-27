import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_3_months_flight_info(date: str, lang: str, arrival: str='false', cargo: str='false', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Today - 90D or Today + 14D"
    date: yyyy-MM-dd
        arrival: true/false
        cargo: true/false
        
    """
    url = f"https://flight-information-of-hong-kong-international-airport.p.rapidapi.com/flightinfo-rest/rest/flights/past"
    querystring = {'date': date, 'lang': lang, }
    if arrival:
        querystring['arrival'] = arrival
    if cargo:
        querystring['cargo'] = cargo
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "flight-information-of-hong-kong-international-airport.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

