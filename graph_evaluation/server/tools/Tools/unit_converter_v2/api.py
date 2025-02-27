import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_abbreviations_for_quantity(quantity: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the abbreviations of units of measure for a given quantity."
    quantity: Quantity to get the units abbreviations for.

Valid quantity names can be retrieved from the  /Quantities endpoint.
        
    """
    url = f"https://unit-converter2.p.rapidapi.com/Quantities/{quantity}/Abbreviations"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unit-converter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def convert(to: str, value: int, is_from: str, quantity: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to do the actual conversion."
    to: Unit of Measure to convert to. Both abbreviations as well as full names can be used, but from and to should be the same.

Valid full unit names can be retrieved from the /Quantities/{quantity}/Units endpoint,
Valid abbreviations can be retrieved from the /Quantities/{quantity}/Abbreviations endpoint,

        value: The value to convert from the 'from' unit of measure to the 'to' unit of measure. Please use a decimal point as the decimal separator.
        from: Unit of Measure to convert from. Both abbreviations as well as full names can be used. 

Valid full unit names can be retrieved from the /Quantities/{quantity}/Units endpoint,
Valid abbreviations can be retrieved from the /Quantities/{quantity}/Abbreviations endpoint,

        quantity: Quantity to convert. 

Valid quantity names can be retrieved from the  /Quantities endpoint.
        
    """
    url = f"https://unit-converter2.p.rapidapi.com/Convert/{quantity}"
    querystring = {'to': to, 'value': value, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unit-converter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_units_for_quantity(quantity: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the full names of units of measure for a given quantity."
    quantity: Quantity to get the units full names for.

Valid quantity names can be retrieved from the  /Quantities endpoint.
        
    """
    url = f"https://unit-converter2.p.rapidapi.com/Quantities/{quantity}/Units"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unit-converter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_quantities(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the list of available quantities."
    
    """
    url = f"https://unit-converter2.p.rapidapi.com/Quantities"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "unit-converter2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

