import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def co2_data_from_and_awb(awb: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get CO2 calculation, and benchmark data per AWB number"
    
    """
    url = f"https://cargo2zero.p.rapidapi.com/co2calculation"
    querystring = {'awb': awb, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cargo2zero.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def co2_data_from_a_flight(weight: str, flightnumber: str, date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get CO2 calculation, and benchmark data from a Flight number and date"
    
    """
    url = f"https://cargo2zero.p.rapidapi.com/co2calculation"
    querystring = {'weight': weight, 'flightnumber': flightnumber, 'date': date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cargo2zero.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def co2_data_per_origin_destination_and_flight(destination: str, aircraft: str, origin: str, weight: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get CO2 calculation and benchmark on route"
    
    """
    url = f"https://cargo2zero.p.rapidapi.com/co2calculation"
    querystring = {'destination': destination, 'aircraft': aircraft, 'origin': origin, 'weight': weight, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cargo2zero.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

