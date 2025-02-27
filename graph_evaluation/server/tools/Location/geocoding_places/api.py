import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_geocoding_with_images(lat: int, lng: int, version: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get complete readable place info with high resolution images in prefered language based on given latitude and longitude."
    lat: Latitude in decimal degrees (wgs84)
        lng: Longitude in decimal degrees (wgs84)
        lang: Prefered language of content.
        
    """
    url = f"https://geocoding-places.p.rapidapi.com/get_geocoding_images/{version}"
    querystring = {'lat': lat, 'lng': lng, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geocoding-places.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_geocoding(version: str, lat: int, lng: int, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get complete readable place info in prefered language based on given coordinate in latitude and longitude."
    lat: Latitude in decimal degrees (wgs84)
        lng: Longitude in decimal degrees (wgs84)
        lang: Prefered language of content.
        
    """
    url = f"https://geocoding-places.p.rapidapi.com/get_geocoding/{version}"
    querystring = {'lat': lat, 'lng': lng, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geocoding-places.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_geocoding_with_videos(lat: int, lng: int, version: str, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get complete readable place info in prefered language with related video on Youtube based on given latitude and longitude."
    lat: Latitude in decimal degrees (wgs84)
        lng: Longitude in decimal degrees (wgs84)
        lang: Prefered language of content.
        
    """
    url = f"https://geocoding-places.p.rapidapi.com/get_geocoding_videos/{version}"
    querystring = {'lat': lat, 'lng': lng, }
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "geocoding-places.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

