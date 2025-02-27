import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_to_all_units(measurement: str, is_from: str, convert: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts the given quantity of a unit to the equivalent amount of all other units of a given measurement that the API provides conversion between. The legal values for measurement can be obtained from the API call GET: dev/measurements. The legal values for the units can be obtained from the API call GET: dev/{measurement}/allUnits."
    
    """
    url = f"https://units-converter.p.rapidapi.com/dev/{measurement}/convertall"
    querystring = {'from': is_from, 'convert': convert, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "units-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_measurements(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of all the measurements for which the API provides conversion"
    
    """
    url = f"https://units-converter.p.rapidapi.com/dev/measurements"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "units-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_the_units_of_a_measurement(measurement: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the units of a measurement that the API provides conversion between. The legal values for measurement can be obtained from the API call GET: dev/measurements."
    
    """
    url = f"https://units-converter.p.rapidapi.com/dev/{measurement}/allUnits"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "units-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_a_quantity_from_one_unit_to_another(measurement: str, is_from: str, convert: int, to: str='kilowatt-hour', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Converts the given quantity in one unit to the equivalent quantity in another, for a given measurement, and returns the answer. The legal values for measurement can be obtained from the API call GET: dev/measurements. The legal values for the units can be obtained from the API call GET: dev/{measurement}/allUnits."
    
    """
    url = f"https://units-converter.p.rapidapi.com/dev/{measurement}"
    querystring = {'from': is_from, 'convert': convert, }
    if to:
        querystring['to'] = to
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "units-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

