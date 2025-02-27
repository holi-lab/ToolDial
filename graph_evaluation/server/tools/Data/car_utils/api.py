import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def estimate_ownership_cost(vin: str, mileage_year: int=15000, mileage_start: int=50000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Estimate how much you will spend on the car with specified VIN for the next 5 years."
    vin: [Vehicle identification number](https://www.autocheck.com/vehiclehistory/vin-basics).
        mileage_year: An estimate of many miles you will drive every year for the next 5 years. Defaults to 15000.
        mileage_start: Start mileage of the car. If unspecified, an estimate will be used.
        
    """
    url = f"https://car-utils.p.rapidapi.com/ownershipcost"
    querystring = {'vin': vin, }
    if mileage_year:
        querystring['mileage_year'] = mileage_year
    if mileage_start:
        querystring['mileage_start'] = mileage_start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-utils.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def estimate_market_value(vin: str, mileage: int=50000, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Estimate market value of the car with specified VIN."
    vin: [Vehicle identification number](https://www.autocheck.com/vehiclehistory/vin-basics).
        mileage: Current mileage of the car.  If unspecified, an estimate will be used.
        
    """
    url = f"https://car-utils.p.rapidapi.com/marketvalue"
    querystring = {'vin': vin, }
    if mileage:
        querystring['mileage'] = mileage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-utils.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def decode_vin(vin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Decode VIN, get valuable information for the car you are interested in."
    vin: [Vehicle identification number](https://www.autocheck.com/vehiclehistory/vin-basics). Incomplete VIN checking is supported.
        
    """
    url = f"https://car-utils.p.rapidapi.com/vindecoder"
    querystring = {'vin': vin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-utils.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fuel_economy_information(model: str, make: str, year: str='2023', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fuel economy information from the official U.S. government source."
    model: The vehicle model. Use /models endpoint to get supported models for the specified make.
        make: The vehicle make. Use /makes endpoint to get supported makes.
        year: Model year. Currently support model years from 1985 to 2023.
        
    """
    url = f"https://car-utils.p.rapidapi.com/fueleconomy"
    querystring = {'model': model, 'make': make, }
    if year:
        querystring['year'] = year
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-utils.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vehicle_models(make: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all supported vehicle models for specified make."
    make: The brand of the vehicle.
        
    """
    url = f"https://car-utils.p.rapidapi.com/models"
    querystring = {'make': make, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-utils.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vehicle_makes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all supported vehicle makes."
    
    """
    url = f"https://car-utils.p.rapidapi.com/makes"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-utils.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

