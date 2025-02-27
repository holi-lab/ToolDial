import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def suburbs(suburb: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for suburbs by name"
    suburb: Part of suburb name or suburb name. Min length 3
        
    """
    url = f"https://australian-suburbs.p.rapidapi.com/dictionaries.php"
    querystring = {'suburb': suburb, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "australian-suburbs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def neighbours(type: str, radius: int, locationid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find neighbouring suburbs based on the distance in km."
    type: Don't change
        radius: Radius around the provided suburb in kilometers (km)
        locationid: Suburb ID
        
    """
    url = f"https://australian-suburbs.p.rapidapi.com/dictionaries.php"
    querystring = {'type': type, 'radius': radius, 'locationId': locationid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "australian-suburbs.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

