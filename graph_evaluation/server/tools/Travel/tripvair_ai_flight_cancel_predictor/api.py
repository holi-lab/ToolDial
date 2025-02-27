import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_flights_v1_cancel(carrier: str, origin: str, departdate: str, dest: str, flightno: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Offers a statistical  probability of a future flight being cancelled using deep learning AI from 7 days up to 363 days in advance"
    carrier: Scheduled airline by code 
        origin: Origin city code for departure
        departdate: Departs 10/29/2023.
        dest: Destination city code for departure
        flightno: Flight Number
        
    """
    url = f"https://tripvair-ai-flight-cancel-predictor.p.rapidapi.com/flights/v1/cancel"
    querystring = {'carrier': carrier, 'origin': origin, 'departdate': departdate, 'dest': dest, 'flightno': flightno, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tripvair-ai-flight-cancel-predictor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

