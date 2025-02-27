import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_vehicle_information(plates: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sample response:
		
		```
		{
		    "doors": 4,
		    "has_stolen_report": false,
		    "year": 2009,
		    "license_plates": "CP50087",
		    "model": "DOBLE CABINA",
		    "assembly_plant_location": "",
		    "state": "CAMPECHE",
		    "classification": "ESTANDAR",
		    "type": "",
		    "make": "NISSAN",
		    "version": "V4",
		    "origin_country": "MEXICO"
		}
		```"
    
    """
    url = f"https://mexican-vehicles-repuve.p.rapidapi.com/"
    querystring = {'plates': plates, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mexican-vehicles-repuve.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

