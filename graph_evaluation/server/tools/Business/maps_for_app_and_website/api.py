import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def zxy_raster_map_tiles(z: str, style: str, x: str, y_and_format: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Vector map tiles to be used in Leaflet, QGIS, and other libraries and GIS.
		**IMPORTANT!** Use ZXY format to add a basemap layer. For example,  `v1/tile/dark-matter-purple-roads/{z}/{x}/{y}.png`"
    z: Zoom
        style: Supported map style: osm-carto and vector
        x: Tile X coordinate
        y_and_format: Tile Y coordinate and format
        
    """
    url = f"https://maps-for-app-and-website.p.rapidapi.com/v1/tile/{style}/{z}/{x}/{y_and_format}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maps-for-app-and-website.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vector_map_tiles(style: str, resource: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Vector map tiles to be used in MapLibre GL (Mapbox GL), OpenLayers.
		**IMPORTANT!** To authorize renderer requests to RapidAPI, you either need to inject "x-rapidapi-key" header into each request issued by renderer OR (easier) add it as the "rapidapi-key" query string parameter."
    style: Supported map style: osm-carto, osm-bright, osm-bright-grey, osm-bright-smooth, klokantech-basic, osm-liberty, maptiler-3d, toner, toner-grey, positron, positron-blue, positron-red, dark-matter, dark-matter-brown, dark-matter-dark-grey, dark-matter-dark-purple, dark-matter-purple-roads, dark-matter-yellow-roads
        resource: Style resources, including style.json, tile.json, fonts, sprites and other meta-resources referenced by a map style.
        
    """
    url = f"https://maps-for-app-and-website.p.rapidapi.com/v1/styles/{style}/{resource}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maps-for-app-and-website.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

