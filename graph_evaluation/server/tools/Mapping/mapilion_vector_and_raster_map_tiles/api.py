import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_vector_tile(x: int, y: int, z: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Vector tiles following the OpenMapTiles schema"
    
    """
    url = f"https://mapilion-vector-and-raster-map-tiles.p.rapidapi.com/rapid-api/vector/{z}/{x}/{y}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapilion-vector-and-raster-map-tiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_vector_tile(x: int, y: int, z: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Vector tiles following the OpenMapTiles schema"
    
    """
    url = f"https://mapilion-vector-and-raster-map-tiles.p.rapidapi.com/rapid-api/vector/{z}/{x}/{y}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapilion-vector-and-raster-map-tiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hillshading_tile(x: int, version: str, z: int, y: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns special hillshading Raster tiles as PNG. Hillshading can be used as an overlay to create the look and feel of mountains. The version can be either v1 or v2."
    
    """
    url = f"https://mapilion-vector-and-raster-map-tiles.p.rapidapi.com/rapid-api/hillshades/{version}/{z}/{x}/{y}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapilion-vector-and-raster-map-tiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_hillshading_tile(x: int, version: str, z: int, y: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns special hillshading Raster tiles as PNG. Hillshading can be used as an overlay to create the look and feel of mountains. The version can be either v1 or v2."
    
    """
    url = f"https://mapilion-vector-and-raster-map-tiles.p.rapidapi.com/rapid-api/hillshades/{version}/{z}/{x}/{y}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapilion-vector-and-raster-map-tiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_tile_json(type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the tile json for the give style. Currently there are three different types:
		- `v3` the OpenMapTiles compatible vector tiles
		- `hillshading-v1` Hillshading tiles Version 1
		- `hillshading-v2` Hillshading tiles Version 2"
    
    """
    url = f"https://mapilion-vector-and-raster-map-tiles.p.rapidapi.com/rapid-api/tilejson/{type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapilion-vector-and-raster-map-tiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fonts(range: str, fontstack: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Fonts for vector tiles. This endpoint is typically used in your style.json."
    
    """
    url = f"https://mapilion-vector-and-raster-map-tiles.p.rapidapi.com/rapid-api/fonts/{fontstack}/{range}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapilion-vector-and-raster-map-tiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_fonts(range: str, fontstack: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Fonts for vector tiles. This endpoint is typically used in your style.json."
    
    """
    url = f"https://mapilion-vector-and-raster-map-tiles.p.rapidapi.com/rapid-api/fonts/{fontstack}/{range}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapilion-vector-and-raster-map-tiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_style_json(style: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the style json for the given style. These are pre-configured styles to get started easily. For advanced use cases we recommend to extend these styles or create a new style. The following styles are currently available:
		- `osm-bright` Osm-Bright
		- `osm-bright-hillshading-v1` Osm-Bright with Hillshading V1 overlay
		- `osm-bright-hillshading-v2` Osm-Bright with Hillshading V2 overlay
		- `dark-matter` A dark map style."
    
    """
    url = f"https://mapilion-vector-and-raster-map-tiles.p.rapidapi.com/rapid-api/stylejson/{style}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapilion-vector-and-raster-map-tiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_raster_tile(x: int, y: int, style_name: str, z: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Raster tiles as PNG. These can be used in mapping libraries like Leaflet. The style-name can be one of:
		- `kurviger-liberty` a map style developed for Kurviger.de
		- `osm-bright` the default Osm Bright style"
    
    """
    url = f"https://mapilion-vector-and-raster-map-tiles.p.rapidapi.com/rapid-api/raster/{style_name}/{z}/{x}/{y}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapilion-vector-and-raster-map-tiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_raster_tile(x: int, y: int, style_name: str, z: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Raster tiles as PNG. These can be used in mapping libraries like Leaflet. The style-name can be one of:
		- `kurviger-liberty` a map style developed for Kurviger.de
		- `osm-bright` the default Osm Bright style"
    
    """
    url = f"https://mapilion-vector-and-raster-map-tiles.p.rapidapi.com/rapid-api/raster/{style_name}/{z}/{x}/{y}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapilion-vector-and-raster-map-tiles.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

