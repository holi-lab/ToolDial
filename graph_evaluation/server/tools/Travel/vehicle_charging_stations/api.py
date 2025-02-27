import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_stations(distance: str='100', longitude: str='2.4379392', latitude: str='44.351488', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return nearest charging stations within
		A POI (Point of Interest), also referred to as a Site or ChargePoint, is the top-level set of information regarding a geographic site with one or more electric vehicle charging equipment present."
    distance: Unit: Miles
        
    """
    url = f"https://vehicle-charging-stations.p.rapidapi.com/poi/"
    querystring = {}
    if distance:
        querystring['distance'] = distance
    if longitude:
        querystring['longitude'] = longitude
    if latitude:
        querystring['latitude'] = latitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "vehicle-charging-stations.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

