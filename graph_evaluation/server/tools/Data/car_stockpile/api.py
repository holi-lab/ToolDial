import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def models_for_make(make: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves all the car models from a given manufacturer/make."
    make: Name of manufacturer/make
        
    """
    url = f"https://car-stockpile.p.rapidapi.com/models"
    querystring = {'make': make, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-stockpile.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def base_trim(year: str, model: str, make: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the ID of the base Trim for a given model. The base trim is the standard trim, i.e. without any extras, typically the lowest priced trim."
    year: First year of manufacture

        model: Name of model
        make: Name of manufacturer/make
        
    """
    url = f"https://car-stockpile.p.rapidapi.com/base-trim"
    querystring = {'year': year, 'model': model, 'make': make, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-stockpile.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def full_specification(model: str, make: str, trim: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the technical specification for a given trim."
    model: Name of model
        make: Name of manufacturer/make
        trim: Name of the specific trim
        year: First year of manufacturing
        
    """
    url = f"https://car-stockpile.p.rapidapi.com/trim-specification"
    querystring = {'model': model, 'make': make, 'trim': trim, 'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-stockpile.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chassis_wheel_info(make: str, trim: str, model: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the chassis and wheel information segment of the specification for a given trim."
    make: Name of manufacturer/make
        trim: Name of the specific trim
        model: Name of model
        year: First year of manufacturing
        
    """
    url = f"https://car-stockpile.p.rapidapi.com/spec-chassis-wheel"
    querystring = {'make': make, 'trim': trim, 'model': model, 'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-stockpile.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def dimensions_info(year: str, make: str, model: str, trim: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the dimensions information segment of the specification for a given trim."
    year: First year of manufacturing
        make: Name of manufacturer/make
        model: Name of model
        trim: Name of the specific trim
        
    """
    url = f"https://car-stockpile.p.rapidapi.com/spec-dimensions"
    querystring = {'year': year, 'make': make, 'model': model, 'trim': trim, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-stockpile.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def electric_engine_info(trim: str, make: str, model: str, year: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the electric engine information segment of the specification for a given trim."
    trim: Name of the specific trim
        make: Name of manufacturer/make
        model: Name of model
        year: First year of manufacturing
        
    """
    url = f"https://car-stockpile.p.rapidapi.com/spec-electric-engine"
    querystring = {'trim': trim, 'make': make, 'model': model, 'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-stockpile.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def features_info(year: str, make: str, trim: str, model: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the features information segment of the specification for a given trim."
    year: First year of manufacturing
        make: Name of manufacturer/make
        trim: Name of the specific trim
        model: Name of model
        
    """
    url = f"https://car-stockpile.p.rapidapi.com/spec-features"
    querystring = {'year': year, 'make': make, 'trim': trim, 'model': model, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-stockpile.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fuel_engine_info(make: str, model: str, year: str, trim: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the fuel engine information segment of the specification for a given trim."
    make: Name of manufacturer/make
        model: Name of model
        year: First year of manufacturing
        trim: Name of the specific trim
        
    """
    url = f"https://car-stockpile.p.rapidapi.com/spec-fuel-engine"
    querystring = {'make': make, 'model': model, 'year': year, 'trim': trim, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-stockpile.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def general_info(make: str, year: str, model: str, trim: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the general information segment of the specification for a given trim."
    make: Name of manufacturer/make
        year: First year of manufacturing
        model: Name of model
        trim: Name of the specific trim
        
    """
    url = f"https://car-stockpile.p.rapidapi.com/spec-general"
    querystring = {'make': make, 'year': year, 'model': model, 'trim': trim, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-stockpile.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def transmission_info(year: str, trim: str, model: str, make: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the transmission information segment of the specification for a given trim."
    year: First year of manufacturing
        trim: Name of the specific trim
        model: Name of model
        make: Name of manufacturer/make
        
    """
    url = f"https://car-stockpile.p.rapidapi.com/spec-transmission"
    querystring = {'year': year, 'trim': trim, 'model': model, 'make': make, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-stockpile.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def models_for_year(year: str, make: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET models for a particular MAKE and YEAR"
    year: First year of manufacturing
        make: Name of manufacturer/make
        
    """
    url = f"https://car-stockpile.p.rapidapi.com/models-for-year"
    querystring = {'year': year, 'make': make, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-stockpile.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trims_for_model(year: str, model: str, make: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve all Trim ID's for a given model."
    year: First year of manufacturing
        model: Name of model
        make: Name of manufacturer/make
        
    """
    url = f"https://car-stockpile.p.rapidapi.com/trims"
    querystring = {'year': year, 'model': model, 'make': make, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "car-stockpile.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

