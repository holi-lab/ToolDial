import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def findplacesnearby(location: str, language: str='en', radius: int=180, type: str='cafe', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for nearby places"
    location: The latitude/longitude around which to retrieve places
        language: The language in which to return results
        radius: The distance (in meters) within which to return results. Max = 10000 m
        type: The type of places that are returned
        
    """
    url = f"https://trueway-places.p.rapidapi.com/FindPlacesNearby"
    querystring = {'location': location, }
    if language:
        querystring['language'] = language
    if radius:
        querystring['radius'] = radius
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trueway-places.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def findplacebytext(text: str, language: str='en', bounds: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for places by text string"
    text: A text string on which to search
        language: The language in which to return results
        bounds: Rectangular bounds (south,west,north,east)
        
    """
    url = f"https://trueway-places.p.rapidapi.com/FindPlaceByText"
    querystring = {'text': text, }
    if language:
        querystring['language'] = language
    if bounds:
        querystring['bounds'] = bounds
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trueway-places.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

