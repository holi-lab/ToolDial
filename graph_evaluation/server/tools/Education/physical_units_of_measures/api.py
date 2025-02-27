import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def units_of_measure(content_type: str, accept: str, unitfrom: str, physical_quantity: str, value: str, unitto: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return the value for converting several different Physical Quantities"
    unitfrom: The unit you want to convert from
        value: the value to be converted
        unitto: The unit you wanto to convert to
        
    """
    url = f"https://physical-units-of-measures.p.rapidapi.com/api/rapi/unitsofmeasure/{physical_quantity}"
    querystring = {'Content-Type': content_type, 'Accept': accept, 'unitFrom': unitfrom, 'value': value, 'unitTo': unitto, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "physical-units-of-measures.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

