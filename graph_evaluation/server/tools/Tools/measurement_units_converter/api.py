import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def convert_from_one_unit_of_measure_to_another(output_unit: str, input_unit: str, value: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Convert efficiently and quickly between more than 50 of the most used units with a simple and intuitive conversion tool. At the output, you will get an answer with the conversion of your measurement units."
    
    """
    url = f"https://measurement-units-converter.p.rapidapi.com/api/v1/market/conversions/convert"
    querystring = {'output_unit': output_unit, 'input_unit': input_unit, 'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "measurement-units-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def measurements(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET array of all types of measurement. The array key is in the format {inputUnit-outputUnit} - you can later use these values ​​when converting units of measurement."
    
    """
    url = f"https://measurement-units-converter.p.rapidapi.com/api/v1/market/conversions/measurements"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "measurement-units-converter.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

