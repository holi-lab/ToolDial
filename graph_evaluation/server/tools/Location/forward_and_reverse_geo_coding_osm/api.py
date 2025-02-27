import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reverse_lookup(lon: int, lat: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "provide latitude and longitude to get the location address and all details"
    
    """
    url = f"https://forward-and-reverse-geo-coding-osm.p.rapidapi.com/reverse.php"
    querystring = {'lon': lon, 'lat': lat, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forward-and-reverse-geo-coding-osm.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_address_or_string(q: str, format: str='json', postalcode: str=None, country: str=None, city: str=None, state: str=None, county: str=None, street: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The search API allows you to look up a location from a textual description or address. Nominatim supports structured and free-form search queries.
		
		The search query may also contain special phrases which are translated into specific OpenStreetMap (OSM) tags (e.g. Pub => amenity=pub). This can be used to narrow down the kind of objects to be returned."
    q: q=<query>
        format: possible values
json, xml, geojson, jsonv2, geocodejson
        
    """
    url = f"https://forward-and-reverse-geo-coding-osm.p.rapidapi.com/search.php"
    querystring = {'q': q, }
    if format:
        querystring['format'] = format
    if postalcode:
        querystring['postalcode'] = postalcode
    if country:
        querystring['country'] = country
    if city:
        querystring['city'] = city
    if state:
        querystring['state'] = state
    if county:
        querystring['county'] = county
    if street:
        querystring['street'] = street
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "forward-and-reverse-geo-coding-osm.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

