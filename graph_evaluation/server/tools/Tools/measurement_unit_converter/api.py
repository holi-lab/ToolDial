import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_from_one_unit_to_another(to: str, is_from: str, value: int, measure: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### Converts the Given Quantity in One Unit to Another
		
		This API call performs the actual unit conversion. You specify the measurement type, the source unit, the target unit, and the value to convert. Additionally, you can use the abbreviation, plural, or singular forms for the query parameters to specify units.
		
		#### Usage 
		Make a GET request to /measurement where <measurement> is the type of measurement (e.g., length, mass). Use query parameters to specify the conversion details:
		
		- **value**: The numeric value you want to convert (e.g., 1200).
		- **from**: The source unit, which can be specified as the abbreviation, singular form, or plural form (e.g., m, meter, meters).
		- **to**: The target unit, which can also be specified as the abbreviation, singular form, or plural form (e.g., km, kilometer, kilometers).
		
		#### Example 1
		To convert 1200 meters to kilometers, you can use any of the following, or you can mix them:
		- from=m, to=km
		- from=meter, to=kilometer
		- from=meters, to=kilometers
		- from=meters, to=kilometer
		- from=m, to=kilometers
		
		#### Example 2
		To convert 5 pounds to ounces, you can use any of the following, or mix them:
		- from=lb, to=oz
		- from=pound, to=ounce
		- from=pounds, to=ounces
		- from=lb, to=ounces
		
		The response will provide the converted value and details.
		
		This allows for flexibility in specifying units in a way that's most convenient for your API users."
    
    """
    url = f"https://measurement-unit-converter.p.rapidapi.com/{measure}"
    querystring = {'to': to, 'from': is_from, 'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "measurement-unit-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def measurements(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### GET Enum Array of All Types of Measurement
		
		This call retrieves an array of all available types of measurements that can be converted using the API."
    
    """
    url = f"https://measurement-unit-converter.p.rapidapi.com/measurements"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "measurement-unit-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def measurements_detailed(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### GET a List of All Available Measurements with Unit Details
		
		This call provides a detailed list of all available measurements along with their unit details, including abbreviations, systems, singular, and plural forms."
    
    """
    url = f"https://measurement-unit-converter.p.rapidapi.com/measurements/detailed"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "measurement-unit-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def measure_units(measure: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "### GET Enum Array of All Units of the Given Type
		
		This call fetches an array of all units associated with a specific measurement type, which you can use for conversions in the fourth API call. 
		
		For example, to get units for length, make a GET request to **/length/units**. The response will contain an array of units you can use for conversions."
    
    """
    url = f"https://measurement-unit-converter.p.rapidapi.com/{measure}/units"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "measurement-unit-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

