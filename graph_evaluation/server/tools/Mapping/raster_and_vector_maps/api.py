import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def map_styles(resource: str, style: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Style definitions and corresponding resources, including layer definitions, fonts, sprites. It is recommended to use "style.json" resource to automatically confiugre OpenLayers, MapboxGL, Leaflet or other map renderers. ----------------------------[IMPORTANT]---------------------------- To authorize renderer requests to RapidAPI, you either need to inject "x-rapidapi-key" header into each request issued by renderer OR (easier) add it as the "rapidapi-key" query string parameter."
    resource: Style resources, including style.json, tile.json, fonts, sprites and other meta-resources referenced by a map style.
        style: Supported map style: osm-carto, osm-bright, klokantech-basic, positron, or dark-matter
        
    """
    url = f"https://raster-and-vector-maps.p.rapidapi.com/v1/styles/{style}/{resource}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "raster-and-vector-maps.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tiles(z: str, x: str, y_and_format: str, style: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Individual map tiles in raster or vector (MVT compatible) format."
    z: Zoom
        x: Tile X coordinate
        y_and_format: Tile Y coordinate and format
        style: Supported map style: osm-carto and vector
        
    """
    url = f"https://raster-and-vector-maps.p.rapidapi.com/v1/tile/{style}/{z}/{x}/{y_and_format}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "raster-and-vector-maps.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

