import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def main(lat2: int, lon1: int, lon2: int, lat1: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return distance in km, true course and magnetic course in deg,"
    lat2: latitude of second point, [-90,90]
        lon1: longitude of first point, [-180,180]
        lon2: longitude of second point, [-180,180]
        lat1: latitude of first point, [-90,90]
        
    """
    url = f"https://course-distance.p.rapidapi.com/course"
    querystring = {'lat2': lat2, 'lon1': lon1, 'lon2': lon2, 'lat1': lat1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "course-distance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

