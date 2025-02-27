import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_by_centre_zip_and_radius(zip: str, radius: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform a single radius search with a defined radius and then a location in the form of a ZIP code OR a latitude/longitude."
    
    """
    url = f"https://usa-zip-codes-inside-radius.p.rapidapi.com/"
    querystring = {'zip': zip, 'radius': radius, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "usa-zip-codes-inside-radius.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_by_latitude_longitude_and_radius(lng: str, radius: int, lat: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Perform a single radius search with a defined radius and then a location in the form of a ZIP code OR a latitude/longitude."
    
    """
    url = f"https://usa-zip-codes-inside-radius.p.rapidapi.com/"
    querystring = {'lng': lng, 'radius': radius, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "usa-zip-codes-inside-radius.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

