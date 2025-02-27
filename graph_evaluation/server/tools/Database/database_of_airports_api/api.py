import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def airports(codeiataairport: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get data about global airports by providing the IATA code."
    codeiataairport: Use this parameter to get information about a specific airport, you can search based on IATA code.
        
    """
    url = f"https://database-of-airports-api.p.rapidapi.com/airports"
    querystring = {'codeIataAirport': codeiataairport, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "database-of-airports-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

