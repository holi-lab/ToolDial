import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_single_phase_power_watts_to_current_ampheres(power: int, powerfactor: int=0, voltage: int=230, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes an input of the following:
		
		- Power in watts - float - required
		- Power factor - float - defaults to 0.95
		- Voltage in volts - float - defaults to 230
		
		The output is in ampheres.
		
		For example an input of the following:
		`?power=3000&voltage=240 `
		
		Would result in:
		`{"current":13.16}`"
    
    """
    url = f"https://electrical-units.p.rapidapi.com/power_to_current/single_phase"
    querystring = {'power': power, }
    if powerfactor:
        querystring['powerfactor'] = powerfactor
    if voltage:
        querystring['voltage'] = voltage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "electrical-units.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_three_phase_power_watts_to_current_ampheres(power: int, powerfactor: int=0, voltage: int=400, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes an input of the following:
		
		- Power in watts - float - required
		- Power factor - float - defaults to 0.95
		- Voltage in volts - float - defaults to 400
		
		The output is in ampheres."
    
    """
    url = f"https://electrical-units.p.rapidapi.com/power_to_current/three_phase"
    querystring = {'power': power, }
    if powerfactor:
        querystring['powerfactor'] = powerfactor
    if voltage:
        querystring['voltage'] = voltage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "electrical-units.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_single_phase_current_ampheres_to_power_watts(current: int, voltage: int=230, powerfactor: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes an input of the following:
		
		- Current in ampheres - float - required
		- Power factor - float - defaults to 0.95
		- Voltage in volts - float - defaults to 230
		
		The output is in watts."
    
    """
    url = f"https://electrical-units.p.rapidapi.com/current_to_power/single_phase"
    querystring = {'current': current, }
    if voltage:
        querystring['voltage'] = voltage
    if powerfactor:
        querystring['powerfactor'] = powerfactor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "electrical-units.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert_three_phase_current_ampheres_to_power_watts(current: int, voltage: int=400, powerfactor: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes an input of the following:
		
		- Current in ampheres - float - required
		- Power factor - float - defaults to 0.95
		- Voltage in volts - float - defaults to 230
		
		The output is in watts."
    
    """
    url = f"https://electrical-units.p.rapidapi.com/current_to_power/three_phase"
    querystring = {'current': current, }
    if voltage:
        querystring['voltage'] = voltage
    if powerfactor:
        querystring['powerfactor'] = powerfactor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "electrical-units.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

