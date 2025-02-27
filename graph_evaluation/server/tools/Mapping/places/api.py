import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def geographic_coordinates_by_placename(name: str, lang: str, country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns geographic coordinates for the given placename (city, village, etc.). The method returns the place whose name is most similar to the search string."
    name: Placename
        lang: Two-letter language code (ISO639-1). The following values are available: en (english), ru (russian)
        country: Two-letter country code, ISO-3166 (optional). Default is all countries.
        
    """
    url = f"https://opentripmap-places-v1.p.rapidapi.com/{lang}/places/geoname"
    querystring = {'name': name, }
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opentripmap-places-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def places_list_by_radius_nearby_search(lang: str, radius: int, lon: int, lat: int, src_attr: str=None, format: str=None, kinds: str=None, src_geom: str=None, rate: str=None, limit: int=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method returns objects closest to the selected point optionally filtered by parameters. Only basic information is include in response: xid, name, kinds, osm, wikidata of each object. Depending on the chosen format, the response is either a simple array of objects (with a smaller volume) or an object in GeoJSON format."
    lang: Language code (2 characters, ISO639-1). The following values are available: en (english), ru (russian)
        radius: Maximum distance from selected point in meters
        lon: Longitude of selected point
        lat: Latitude of selected point
        src_attr: The source of the object attributes. It is allowed to point multiple sources separated by commas. Objects from all sources are returned by default. Available values : osm, wikidata, snow, cultura.ru, rosnedra, user
        format: The output format (GeoJSON is set by default). Specify “count” to get the number of obtained objects. Available values : json, geojson, count
        kinds: Object category. Several comma-separated categories may be stated with OR logic. Objects from all categories are returned by default. See object category hierarchy at https://dev.opentripmap.com/doc/en/
        src_geom: The source of the object geometry. Objects from all sources are returned by default. Available values : osm, wikidata, snow, cultura.ru, rosnedra
        rate: Minimum rating of the object popularity, 1 - minimum, 3- maximum, h - object is referred to the cultural heritage. Objects from all categories are returned by default.  Available values : 1, 2, 3, 1h, 2h, 3h
        limit: Maximum number of returned objects. 500 is set by default.
        name: The text string on which to search at the begining of the object name (mimimum 3 characters). All objects are returned by default.
        
    """
    url = f"https://opentripmap-places-v1.p.rapidapi.com/{lang}/places/radius"
    querystring = {'radius': radius, 'lon': lon, 'lat': lat, }
    if src_attr:
        querystring['src_attr'] = src_attr
    if format:
        querystring['format'] = format
    if kinds:
        querystring['kinds'] = kinds
    if src_geom:
        querystring['src_geom'] = src_geom
    if rate:
        querystring['rate'] = rate
    if limit:
        querystring['limit'] = limit
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opentripmap-places-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def places_list_by_bounding_box(lon_max: int, lat_min: int, lang: str, lon_min: int, lat_max: int, name: str=None, format: str=None, limit: int=None, src_geom: str=None, src_attr: str=None, kinds: str=None, rate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method returns all objects (or number of objects) in the given boundary box optionally filtered by parameters. Only basic information is include in response: xid, name, kinds, osm, wikidata of each object. Depending on the chosen format, the response is either a simple array of objects (with a smaller volume) or an object in GeoJSON format."
    lon_max: Maximum longitude
        lat_min: Minimum latitude
        lang: Language code (2 characters, ISO639-1). The following values are available: en (english), ru (russian)
        lon_min: Minimum longitude
        lat_max: Maximum latitude
        name: The text string on which to search at the begining of the object name (mimimum 3 characters). All objects are returned by default.
        format: The output format (GeoJSON is set by default). Specify “count” to get the number of obtained objects. Available values : json, geojson, count
        limit: Maximum number of returned objects. 500 is set by default.
        src_geom: The source of the object geometry. Objects from all sources are returned by default. Available values : osm, wikidata, snow, cultura.ru, rosnedra
        src_attr: The source of the object attributes. It is allowed to point multiple sources separated by commas. Objects from all sources are returned by default. Available values : osm, wikidata, snow, cultura.ru, rosnedra, user
        kinds: Object category. Several comma-separated categories may be stated with OR logic. Objects from all categories are returned by default. See object category hierarchy at https://dev.opentripmap.com/doc/en/
        rate: Minimum rating of the object popularity, 1 - minimum, 3- maximum, h - object is referred to the cultural heritage. Objects from all categories are returned by default.  Available values : 1, 2, 3, 1h, 2h, 3h
        
    """
    url = f"https://opentripmap-places-v1.p.rapidapi.com/{lang}/places/bbox"
    querystring = {'lon_max': lon_max, 'lat_min': lat_min, 'lon_min': lon_min, 'lat_max': lat_max, }
    if name:
        querystring['name'] = name
    if format:
        querystring['format'] = format
    if limit:
        querystring['limit'] = limit
    if src_geom:
        querystring['src_geom'] = src_geom
    if src_attr:
        querystring['src_attr'] = src_attr
    if kinds:
        querystring['kinds'] = kinds
    if rate:
        querystring['rate'] = rate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opentripmap-places-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def place_properties(xid: str, lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns detailed information about the object. Objects can contain different amount of information."
    xid: Unique identifier of the object in OpenTripMap
        lang: Two-letter language code (ISO639-1). The following values are available: en (english), ru (russian)
        
    """
    url = f"https://opentripmap-places-v1.p.rapidapi.com/{lang}/places/xid/{xid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opentripmap-places-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def autosuggest(name: str, lon: int, radius: int, lang: str, lat: int, kinds: str='foods', rate: str=None, src_attr: str=None, format: str=None, limit: int=10, src_geom: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Method returns suggestions for search term closest to the selected point optionally filtered by parameters. Only basic information is include in response: xid, name, kinds, osm, wikidata of each object. Depending on the chosen format, the response is either a simple array of objects (with a smaller volume) or an object in GeoJSON format."
    name: The query term on which to search.
        lon: Longitude of selected point
        radius: Maximum distance from selected point in meters
        lang: Language code (2 characters, ISO639-1). The following values are available: en (english), ru (russian)
        lat: Latitude of selected point
        kinds: Object category. Several comma-separated categories may be stated with OR logic. Objects from all categories are returned by default. See object category hierarchy at https://dev.opentripmap.com/doc/en/
        rate: Minimum rating of the object popularity, 1 - minimum, 3- maximum, h - object is referred to the cultural heritage. Objects from all categories are returned by default.  Available values : 1, 2, 3, 1h, 2h, 3h
        src_attr: The source of the object attributes. It is allowed to point multiple sources separated by commas. Objects from all sources are returned by default. Available values : osm, wikidata, snow, cultura.ru, rosnedra, user
        format: The output format (GeoJSON is set by default). Specify “count” to get the number of obtained objects. Available values : json, geojson
        limit: Maximum number of returned objects. 500 is set by default.
        src_geom: The source of the object geometry. Objects from all sources are returned by default. Available values : osm, wikidata, snow, cultura.ru, rosnedra
        
    """
    url = f"https://opentripmap-places-v1.p.rapidapi.com/{lang}/places/autosuggest"
    querystring = {'name': name, 'lon': lon, 'radius': radius, 'lat': lat, }
    if kinds:
        querystring['kinds'] = kinds
    if rate:
        querystring['rate'] = rate
    if src_attr:
        querystring['src_attr'] = src_attr
    if format:
        querystring['format'] = format
    if limit:
        querystring['limit'] = limit
    if src_geom:
        querystring['src_geom'] = src_geom
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "opentripmap-places-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

