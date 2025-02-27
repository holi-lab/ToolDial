import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def autocomplete_places(limit: int=10, skip: int=0, language: str=None, country: str='US,CA', admindivision1: str=None, accuracyradiuskm: int=None, latitude: int=None, admindivision2: str=None, q: str='Sea', longitude: int=None, type: str='CITY', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of places matching a prefix and specified filter properties. Useful to create "search as you type" inputs."
    limit: Maximum number of places to return. Used together with \"skip\" to paginate results. Defaults to 10, maximum is 100.
        skip: Amount of places to ignore before beginning to return results. Used together with \"limit\" to paginate results. Defaults to 0.
        language: Specifies a language (ISO 639-1) to get the localized name of the place. If translation is not available, \"localizedName\" property will be null.
        country: Filters places by their country \"id\". It's possible to specify multiple values separating them with commas (Ex. ?country=US,CA,MX).
        admindivision1: Filters places by their adminDivision1 \"id\". It's possible to specify multiple values separating them with commas (Ex. ?country=US.CA,US.DE).
        accuracyradiuskm: Maximum radius from the point specified by \"latitude\" and \"longitude\" to filter places located within the area. The value must be expressed in Kilometers. Defaults to 100km.
        latitude: Latitude component of a coordinates set to filter places by their location. This parameter is ignored if \"longitude\" is not specified.
        admindivision2: Filters places by their adminDivision2 \"id\". It's possible to specify multiple values separating them with commas.
        q: Query string to find places which name starts with this prefix.
        longitude: Longitude component of a coordinates set to filter places by their location. This parameter is ignored if \"latitude\" is not specified.
        type: Filters places by their \"type\". It's possible to specify multiple values separating them with commas. Valid types are CITY, ADMIN_DIVISION_1, ADMIN_DIVISION_2 and COUNTRY.
        
    """
    url = f"https://spott.p.rapidapi.com/places/autocomplete"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if skip:
        querystring['skip'] = skip
    if language:
        querystring['language'] = language
    if country:
        querystring['country'] = country
    if admindivision1:
        querystring['adminDivision1'] = admindivision1
    if accuracyradiuskm:
        querystring['accuracyRadiusKm'] = accuracyradiuskm
    if latitude:
        querystring['latitude'] = latitude
    if admindivision2:
        querystring['adminDivision2'] = admindivision2
    if q:
        querystring['q'] = q
    if longitude:
        querystring['longitude'] = longitude
    if type:
        querystring['type'] = type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spott.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_places(type: str='CITY', language: str=None, skip: int=0, country: str='US,CA', limit: int=10, admindivision1: str=None, admindivision2: str=None, accuracyradiuskm: int=None, latitude: int=None, q: str='New York', longitude: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of places (either countries, cities or administrative divisions) matching a query and filtered by properties."
    type: Filters places by \"type\". It's possible to specify multiple values separating them with commas. Valid types are CITY, ADMIN_DIVISION_1, ADMIN_DIVISION_2 and COUNTRY.
        language: Specifies a language (ISO 639-1) to get the localized name of the place. If translation is not available, "localizedName" property will be null.
        skip: Amount of places to ignore before beginning to return results. Used together with \"limit\" to paginate results. Defaults to 0.
        country: Filters places by their country \"id\". It's possible to specify multiple values separating them with commas (Ex. ?country=US,CA,MX).
        limit: Maximum number of places to return. Used together with \"skip\" to paginate results. Defaults to 10, maximum is 100.
        admindivision1: Filters places by their adminDivision1 \"id\". It's possible to specify multiple values separating them with commas (Ex. ?country=US.CA,US.DE).
        admindivision2: Filters places by their adminDivision2 \"id\". It's possible to specify multiple values separating them with commas.
        accuracyradiuskm: Maximum radius from the point specified by \"latitude\" and \"longitude\" to filter places located within the area. The value must be expressed in Kilometers. Defaults to 100km.
        latitude: Latitude component of a coordinates set to filter places by their location. This parameter is ignored if \"longitude\" is not specified.
        q: Query string to find places with a similar name.
        longitude: Longitude component of a coordinates set to filter places by their location. This parameter is ignored if \"latitude\" is not specified.
        
    """
    url = f"https://spott.p.rapidapi.com/places"
    querystring = {}
    if type:
        querystring['type'] = type
    if language:
        querystring['language'] = language
    if skip:
        querystring['skip'] = skip
    if country:
        querystring['country'] = country
    if limit:
        querystring['limit'] = limit
    if admindivision1:
        querystring['adminDivision1'] = admindivision1
    if admindivision2:
        querystring['adminDivision2'] = admindivision2
    if accuracyradiuskm:
        querystring['accuracyRadiusKm'] = accuracyradiuskm
    if latitude:
        querystring['latitude'] = latitude
    if q:
        querystring['q'] = q
    if longitude:
        querystring['longitude'] = longitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spott.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_place_by_ip(is_id: str, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the Place where a given IP Address is located. Returns "Not Found" error when no place is related to the IP. When sending '127.0.0.1' or '0.0.0.0' IP Addresses it will return the Place from the request was performed."
    id: IP Address (v4 and v6 are supported).
        language: Specifies a language (ISO 639-1) to get the localized name of the place. If translation is not available, "localizedName" property will be null.
        
    """
    url = f"https://spott.p.rapidapi.com/places/ip/{is_id}"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spott.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_place_by_my_ip(language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the place related to the IP where the request was performed. Returns "Not Found" error when no place is related to the IP."
    language: Specifies a language (ISO 639-1) to get the localized name of the place. If translation is not available, "localizedName" property will be null.
        
    """
    url = f"https://spott.p.rapidapi.com/places/ip/me"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spott.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_place_by_id(is_id: str, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single Place identified by an ID."
    id: ID of the Place.
        language: Specifies a language (ISO 639-1) to get the localized name of the place. If translation is not available, "localizedName" property will be null.
        
    """
    url = f"https://spott.p.rapidapi.com/places/{is_id}"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spott.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_place_by_geoname_id(geonameid: int, language: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single Place identified by a Geoname ID."
    geonameid: Unique identificator given by Geonames
        language: Specifies a language (ISO 639-1) to get the localized name of the place. If translation is not available, \"localizedName\" property will be null.
        
    """
    url = f"https://spott.p.rapidapi.com/places/geoname-id/{geonameid}"
    querystring = {}
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "spott.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

