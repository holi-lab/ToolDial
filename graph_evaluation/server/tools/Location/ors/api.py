import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def places(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Points of Interest in the area surrounding a point or a line geometry. Use the post endpoint for requests that exceed the get character limit."
    
    """
    url = f"https://orsapi.p.rapidapi.com/places"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "orsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def directions(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a route between two or more locations for a selected profile and its settings as GeoJSON response."
    
    """
    url = f"https://orsapi.p.rapidapi.com/directions"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "orsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geocoding(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint can be used for geocoding (specified `query`) and reverse geocoding requests (specified `location`). Either `query` or `location` has to be specified for a valid request.. If both parameters are specified `location` takes precedence.  - geocoding: Returns a JSON formatted list of objects corresponding to the search input. - reverse geocoding: Returns the next enclosing object with an address tag which surrounds the given coordinate."
    
    """
    url = f"https://orsapi.p.rapidapi.com/geocoding"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "orsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def isochrones(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Isochrone Service supports time and distance analyses for one single or multiple locations. You may also specify the isochrone interval or provide multiple exact isochrone range values. This service allows the same range of profile options listed in the ORS Routing section which help you to further customize your request to obtain a more detailed reachability area response."
    
    """
    url = f"https://orsapi.p.rapidapi.com/isochrones"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "orsapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

