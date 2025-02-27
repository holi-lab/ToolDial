import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def forwardgeocoding(json_callback: str=None, polygon_kml: str=None, bounded: str=None, addressdetails: str=None, city: str='New York City', namedetails: str=None, countrycodes: str=None, limit: str=None, accept_language: str='en', format: str=None, postalcode: str='10011', country: str='USA', county: str=None, state: str='NY', street: str='34 West 13th Street', viewbox: str=None, polygon_text: str=None, polygon_geojson: str=None, polygon_threshold: int=0, polygon_svg: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Turn an address into latitude and longitude (e.g. to display on a map) by schematic input."
    json_callback: Use this with format=json to define a callback function for jsonp. 
        bounded: Used in conjunction with viewbox to search only in an area.
        namedetails: - namedetails = [0,1]

Set this to 1 to receive different spellings in different languages of a returned place.
Default is 0.
        countrycodes: Two letter country code to restrict area to search in. Use the country code as defined in ISO 3166-1 alpha2.
        accept_language: Set the query and response language. Accepts two letter language codes (e.g. 'en' or 'es') or a combination of language and region code, e.g. 'en-GB' or 'es-AR'. Default: 'en'.
        viewbox: Format: x1,y1,x2,y2 - restrict search area to a bounded box defined by two coordinates (x is longitude, y is latitude). The two coordinates have to span a box to work.
        
    """
    url = f"https://forward-reverse-geocoding.p.rapidapi.com/v1/forward"
    querystring = {}
    if json_callback:
        querystring['json_callback'] = json_callback
    if polygon_kml:
        querystring['polygon_kml'] = polygon_kml
    if bounded:
        querystring['bounded'] = bounded
    if addressdetails:
        querystring['addressdetails'] = addressdetails
    if city:
        querystring['city'] = city
    if namedetails:
        querystring['namedetails'] = namedetails
    if countrycodes:
        querystring['countrycodes'] = countrycodes
    if limit:
        querystring['limit'] = limit
    if accept_language:
        querystring['accept-language'] = accept_language
    if format:
        querystring['format'] = format
    if postalcode:
        querystring['postalcode'] = postalcode
    if country:
        querystring['country'] = country
    if county:
        querystring['county'] = county
    if state:
        querystring['state'] = state
    if street:
        querystring['street'] = street
    if viewbox:
        querystring['viewbox'] = viewbox
    if polygon_text:
        querystring['polygon_text'] = polygon_text
    if polygon_geojson:
        querystring['polygon_geojson'] = polygon_geojson
    if polygon_threshold:
        querystring['polygon_threshold'] = polygon_threshold
    if polygon_svg:
        querystring['polygon_svg'] = polygon_svg
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forward-reverse-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reversegeocoding(lon: int, lat: int, json_callback: str=None, polygon_geojson: str=None, limit: str=None, accept_language: str='en', polygon_kml: str=None, zoom: str=None, polygon_svg: str=None, addressdetails: str=None, namedetails: str=None, polygon_threshold: int=0, format: str=None, polygon_text: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find address or place by latitude and longitude"
    accept_language: Set the response language. Accepts two letter language codes (e.g. 'en' or 'es') or a combination of language and region code, e.g. 'en-GB' or 'es-AR'. Default: 'en'.
        
    """
    url = f"https://forward-reverse-geocoding.p.rapidapi.com/v1/reverse"
    querystring = {'lon': lon, 'lat': lat, }
    if json_callback:
        querystring['json_callback'] = json_callback
    if polygon_geojson:
        querystring['polygon_geojson'] = polygon_geojson
    if limit:
        querystring['limit'] = limit
    if accept_language:
        querystring['accept-language'] = accept_language
    if polygon_kml:
        querystring['polygon_kml'] = polygon_kml
    if zoom:
        querystring['zoom'] = zoom
    if polygon_svg:
        querystring['polygon_svg'] = polygon_svg
    if addressdetails:
        querystring['addressdetails'] = addressdetails
    if namedetails:
        querystring['namedetails'] = namedetails
    if polygon_threshold:
        querystring['polygon_threshold'] = polygon_threshold
    if format:
        querystring['format'] = format
    if polygon_text:
        querystring['polygon_text'] = polygon_text
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forward-reverse-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geocodingsearch(q: str, countrycodes: str=None, json_callback: str=None, polygon_text: str=None, namedetails: str=None, limit: str=None, viewbox: str=None, format: str=None, polygon_geojson: str=None, bounded: str=None, polygon_svg: str=None, polygon_kml: str=None, polygon_threshold: int=0, accept_language: str='en', addressdetails: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Free-form query forward geocoding."
    countrycodes: Two-letter country code as defined in ISO 3166-1 alpha2 to restrict search to a country.
        json_callback: Name of json callback function for jsonp calls. Use format=json for this.
        viewbox: Format: x1,y1,x2,y2 where is x is longitude and y latitude. The two coordinates have to span a box to let this work in conjunction with bounded=1 (see there).
        bounded: Use bounded=1 in conjunction with viewbox to restrict search area. You can use this to search for amenities in a given area, e.g. post offices or police stations, etc. as q param.
        accept_language: Set the query and response language. Accepts two letter language codes (e.g. 'en' or 'es') or a combination of language and region code, e.g. 'en-GB' or 'es-AR'. Default: 'en'.
        
    """
    url = f"https://forward-reverse-geocoding.p.rapidapi.com/v1/search"
    querystring = {'q': q, }
    if countrycodes:
        querystring['countrycodes'] = countrycodes
    if json_callback:
        querystring['json_callback'] = json_callback
    if polygon_text:
        querystring['polygon_text'] = polygon_text
    if namedetails:
        querystring['namedetails'] = namedetails
    if limit:
        querystring['limit'] = limit
    if viewbox:
        querystring['viewbox'] = viewbox
    if format:
        querystring['format'] = format
    if polygon_geojson:
        querystring['polygon_geojson'] = polygon_geojson
    if bounded:
        querystring['bounded'] = bounded
    if polygon_svg:
        querystring['polygon_svg'] = polygon_svg
    if polygon_kml:
        querystring['polygon_kml'] = polygon_kml
    if polygon_threshold:
        querystring['polygon_threshold'] = polygon_threshold
    if accept_language:
        querystring['accept-language'] = accept_language
    if addressdetails:
        querystring['addressdetails'] = addressdetails
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forward-reverse-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

