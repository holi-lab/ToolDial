import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def routing_enhancement(kml: str='forward geometry K-UrlML', geometry: str='forward geometry GeoJSON', gpx_backward: str='backward geometry GPX-Url', kml_backward: str='backward geometry KML-Url', routetype: str='bike', callback: str='JSONP Callback Functionname', gpx: str='forward geometry GPX-Url', end: str='lng,lat', start: str='lng,lat', geometry_backward: str='backward geometry GeoJSON', language: str='de', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create a routing along a specific cycleway"
    
    """
    url = f"https://maptoolkit.p.rapidapi.com/enhance/routing"
    querystring = {}
    if kml:
        querystring['kml'] = kml
    if geometry:
        querystring['geometry'] = geometry
    if gpx_backward:
        querystring['gpx_backward'] = gpx_backward
    if kml_backward:
        querystring['kml_backward'] = kml_backward
    if routetype:
        querystring['routeType'] = routetype
    if callback:
        querystring['callback'] = callback
    if gpx:
        querystring['gpx'] = gpx
    if end:
        querystring['end'] = end
    if start:
        querystring['start'] = start
    if geometry_backward:
        querystring['geometry_backward'] = geometry_backward
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptoolkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def route_enhancement(elevation: bool=None, cache: bool=None, gpx: str='GPX Fileurl', routetype: str='bike', geometry: str='GeoJSON geometry string', language: str='de', surface: bool=None, kml: str='KML Fileurl', mapmatch: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enhance your GPX-file with elevation, surface and mapmatching"
    
    """
    url = f"https://maptoolkit.p.rapidapi.com/enhance/route"
    querystring = {}
    if elevation:
        querystring['elevation'] = elevation
    if cache:
        querystring['cache'] = cache
    if gpx:
        querystring['gpx'] = gpx
    if routetype:
        querystring['routeType'] = routetype
    if geometry:
        querystring['geometry'] = geometry
    if language:
        querystring['language'] = language
    if surface:
        querystring['surface'] = surface
    if kml:
        querystring['kml'] = kml
    if mapmatch:
        querystring['mapmatch'] = mapmatch
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptoolkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def global_terrain_raster(z: str, x: str, y: str, ratio: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Global Terrain Rastertiles."
    ratio: Set to `2` for retina tiles.
        
    """
    url = f"https://maptoolkit.p.rapidapi.com/tiles/{z}/{x}/{y}/terrain.png"
    querystring = {}
    if ratio:
        querystring['ratio'] = ratio
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptoolkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def global_terrain_winter(x: str, z: str, y: str, ratio: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Global Terrain Tiles with winter colors."
    ratio: Set to `2` for retina tiles.
        
    """
    url = f"https://maptoolkit.p.rapidapi.com/tiles/{z}/{x}/{y}/terrainwinter.png"
    querystring = {}
    if ratio:
        querystring['ratio'] = ratio
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptoolkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reverse(lon: str, lat: str, polygon: str=None, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reverse geocoding."
    polygon: Language for the given instructions. Must be a valid ISO 639-1 language code. Default value is `de`.
        language: Language for the given instructions. Must be a valid ISO 639-1 language code. Default value is `de`.
        
    """
    url = f"https://maptoolkit.p.rapidapi.com/geocode/reverse"
    querystring = {'lon': lon, 'lat': lat, }
    if polygon:
        querystring['polygon'] = polygon
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptoolkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def staticmaps(size: str, center: str, zoom: int, bounds: str=None, geojson: str=None, maptype: str='toursprung-terrain', path: str=None, delta_zoom: int=None, kml: str=None, format: str='png', marker: str=None, factor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create a Staticmap."
    bounds: Will be used instead of `center` and `zoom` if defined. Format is `{north},{east},{south},{west}`.
        geojson: Value: 
`{attribute}:{value}[|{attribute}:{value} ...]`
Attributes:
`geometry` is a GeoJSON string of type `LineString` or `MultiLineString`
`color` in hex color format `{AABBGGRR}`
`width` sets the lines thickness

Repeated `geojson` parameters supported.
        maptype: Defines the maps appearence.
        path: Value: 
`{attribute}:{value}[|{attribute}:{value} ...]`
Attributes:
`points` are the paths coordinates `{lat},{lng}[|{lat}:{lng} ...]`
`color` in hex color format `{AABBGGRR}`
`width` sets the lines thickness

Repeated `path` parameters supported.
        delta_zoom: Defines how many zoom levels will get added/removed if it the `zoom` was calculated automatically.
        kml: Value: 
`{kml}?{attribute}={value}[&{attribute}={value} ...]`
Attributes:
`color` in hex color format `{AABBGGRR}`
`width` sets the lines thickness

Repeated `kml` parameters supported.
        marker: Value: 
`{attribute}:{value}[|{attribute}:{value} ...]`
Attributes:
`center` is the markers location
`icon` can be any public URL of a PNG or JPEG
`shadow` can be set to `false` to change the markers anchor from center to bottom

Repeated `marker` parameters supported.


        factor: Defines the images scaling factor.
        
    """
    url = f"https://maptoolkit.p.rapidapi.com/staticmap"
    querystring = {'size': size, 'center': center, 'zoom': zoom, }
    if bounds:
        querystring['bounds'] = bounds
    if geojson:
        querystring['geojson'] = geojson
    if maptype:
        querystring['maptype'] = maptype
    if path:
        querystring['path'] = path
    if delta_zoom:
        querystring['delta_zoom'] = delta_zoom
    if kml:
        querystring['kml'] = kml
    if format:
        querystring['format'] = format
    if marker:
        querystring['marker'] = marker
    if factor:
        querystring['factor'] = factor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptoolkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def elevation(points: str, simplify: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Global elevation service."
    points: JSON Array of [latitdue, longitude] Objects.
        simplify: Simplifies the points before searching for elevation.
        
    """
    url = f"https://maptoolkit.p.rapidapi.com/elevation"
    querystring = {'points': points, }
    if simplify:
        querystring['simplify'] = simplify
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptoolkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def routing(points: str, finish_instruction: str=None, language: str=None, voice_instructions: str=None, filename: str=None, format: str=None, weighting: str=None, routetype: str='bike', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Global routing engine."
    points: Value: `{lat},{lng}[|{lat},{lng} ...]`
Define multiple routing waypoints, consisting of latitude and longitude. At least 2 points are needed.
        finish_instruction: Adds a finish instruction at the end of the route.
        language: Language for the given instructions. Must be a valid ISO 639-1 language code. Default value is `en`.
        voice_instructions: Enables voice instructions for text to speech engines.
        filename: Only available if the set `format` is `gpx` or `kml`. 
        format: Default format is `json`.
        weighting: Only available if the set `routeType` is `bike`. Default value is `networks`.
        
    """
    url = f"https://maptoolkit.p.rapidapi.com/route"
    querystring = {'points': points, }
    if finish_instruction:
        querystring['finish_instruction'] = finish_instruction
    if language:
        querystring['language'] = language
    if voice_instructions:
        querystring['voice_instructions'] = voice_instructions
    if filename:
        querystring['filename'] = filename
    if format:
        querystring['format'] = format
    if weighting:
        querystring['weighting'] = weighting
    if routetype:
        querystring['routeType'] = routetype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptoolkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, language: str=None, viewbox: str=None, polygon: str=None, limit: int=None, countrycodes: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Global Geocoder."
    language: Language for the given instructions. Must be a valid ISO 639-1 language code. Default value is `de`.
        viewbox: Defines a preferred area to search in. Format is `{minLng},{minLat},{maxLng},{maxLat}`.
        polygon: Defines whether available polygons are added to the response data or not. Default value is `0`.
        limit: Limits the number of returned results. Default value is `10`.
        countrycodes: Limits result to one or multiple countries. Passing multiple countries, they need to be separated by a comma `,`. Must be a valid  ISO 3166-1 alpha-2 country code.
        
    """
    url = f"https://maptoolkit.p.rapidapi.com/geocode/search"
    querystring = {'q': q, }
    if language:
        querystring['language'] = language
    if viewbox:
        querystring['viewbox'] = viewbox
    if polygon:
        querystring['polygon'] = polygon
    if limit:
        querystring['limit'] = limit
    if countrycodes:
        querystring['countrycodes'] = countrycodes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptoolkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contours_vector_tiles(x: str, z: str, y: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Global Contourlines."
    
    """
    url = f"https://maptoolkit.p.rapidapi.com/tiles/{z}/{x}/{y}/contours.pbf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptoolkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def planet_vector_tiles(y: str, x: str, z: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Global planet vector tiles."
    
    """
    url = f"https://maptoolkit.p.rapidapi.com/tiles/{z}/{x}/{y}/planet.pbf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptoolkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def planet_contour_vectortiles(y: str, z: str, x: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Global Vectortiles combined with Contourlines."
    
    """
    url = f"https://maptoolkit.p.rapidapi.com/tiles/{z}/{x}/{y}/planet-contours.pbf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptoolkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def terrain_rgb(x: str, y: str, z: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Global TerrainRGB Tiles."
    
    """
    url = f"https://maptoolkit.p.rapidapi.com/tiles/{z}/{x}/{y}/terrainrgb.webp"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptoolkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hillshading(x: str, y: str, z: str, ratio: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Global raster hillshading tiles."
    ratio: Set to `2` for retina tiles.
        
    """
    url = f"https://maptoolkit.p.rapidapi.com/tiles/{z}/{x}/{y}/hillshading.png"
    querystring = {}
    if ratio:
        querystring['ratio'] = ratio
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptoolkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def global_light_tiles(x: str, y: str, z: str, ratio: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Global mostly grayscale tiles."
    ratio: Set to `2` for retina tiles.
        
    """
    url = f"https://maptoolkit.p.rapidapi.com/tiles/{z}/{x}/{y}/light.png"
    querystring = {}
    if ratio:
        querystring['ratio'] = ratio
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "maptoolkit.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

