import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def place_search(radius: str, ll: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for places in the FSQ Places database using a location and querying by name, category name, taste label, or chain name. For example, search for "coffee" to get back a list of recommended coffee shops.
		
		You must pass a location with your request by using one of the following options:
		
		ll & radius (circular boundary)
		near (geocodable locality)
		ne & sw (rectangular boundary)"
    radius: Defines the distance (in meters) within which to bias place results. The maximum allowed radius is 100,000 meters.


        ll: The latitude/longitude around which to retrieve place information. This must be specified as latitude,longitude (e.g., ll=41.8781,-87.6298). One of ll, ne and sw, or near must be specified.


        
    """
    url = f"https://hk4.p.rapidapi.com/places/search"
    querystring = {'radius': radius, 'll': ll, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hk4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

