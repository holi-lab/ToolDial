import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def v1_airlines(iata: str=None, name: str='Singapore Airlines', icao: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Ninjas Airlines API endpoint. At least one of the optional parameters must be provided."
    iata: International Air Transport Association (IATA) 2-character airline code.
        name: Airline name. This parameter supports partial matching (e.g. United will match United Airlines)
        icao: International Civil Aviation Organization (ICAO) 3-character airline code.
        
    """
    url = f"https://airlines-by-api-ninjas.p.rapidapi.com/v1/airlines"
    querystring = {}
    if iata:
        querystring['iata'] = iata
    if name:
        querystring['name'] = name
    if icao:
        querystring['icao'] = icao
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airlines-by-api-ninjas.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

