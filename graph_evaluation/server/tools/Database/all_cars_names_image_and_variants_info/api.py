import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_all_cars(vehicle_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API will fetch you all cars list, 
		If you need bike details change **vehicle_type** from "car" to "bike"."
    
    """
    url = f"https://all-cars-names-image-and-variants-info.p.rapidapi.com/motororchestrator/api/v1/mmv"
    querystring = {'vehicle_type': vehicle_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "all-cars-names-image-and-variants-info.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

