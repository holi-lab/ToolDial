import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def zipcodessearchinboundingbox(northeast: str, southwest: str, intersect: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ZIP Codes boundaries in provided bounding box (in GeoJSON format). For a quick view, copy and paste results: http://geojsonlint.com"
    northeast: Coordinates of north-east corner of bounding box. Test example: 40.81,-73.41
        southwest: Coordinates of south-west corner of bounding box. Test example: 40.62,-74.73
        intersect: Include areas that intersect with bounding box
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/zipcodes/search/in-bounding-box"
    querystring = {'northEast': northeast, 'southWest': southwest, }
    if intersect:
        querystring['intersect'] = intersect
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zipcodeslistbystate(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a plain list of ZIP Codes of a state. Result is a plain list of ZIP Codes, not GeoJSON format. To retrieve boundaries of all ZIP Codes of a state you may query method "zipCodesByIds" with a portion of current method output."
    state: State name abbreviation. Test example: NY
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/zipcodes/by-state/list"
    querystring = {'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zipcodessearchinradius(longitude: int, latitude: int, radius: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ZIP Codes boundaries in circle with given radius (in GeoJSON format). For a quick view, copy and paste results: http://geojsonlint.com"
    longitude: Longitude of radius center. Test example: -74.72
        latitude: Latitude of radius center. Test example: 40.61
        radius: Radius size in miles. Min value is 1, max value is 50 miles.
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/zipcodes/search/in-radius"
    querystring = {'longitude': longitude, 'latitude': latitude, 'radius': radius, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countiesbyids(ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get boundaries of given Counties (in GeoJSON format). For a quick view, copy and paste results: http://geojsonlint.com"
    ids: Array of comma separated county ids. Test example: 20109,20111,20113
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/counties/by-ids"
    querystring = {'ids': ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countybynameandstate(name: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get boundaries of County by name and state (in GeoJSON format).  For a quick view, copy and paste results: http://geojsonlint.com"
    name: Full county name. Test example: Logan
        state: State name abbreviation. Test example: KS
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/counties/by-name-and-state"
    querystring = {'name': name, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countiesbystate(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Counties boundaries of state (in GeoJSON format). For a quick view, copy and paste results: http://geojsonlint.com"
    state: State name abbreviation. Test example: TX
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/counties/by-state"
    querystring = {'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countiessearchinboundingbox(northeast: str, southwest: str, intersect: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Counties boundaries in provided bounding box (in GeoJSON format). For a quick view, copy and paste results: http://geojsonlint.com"
    northeast: Coordinates of north-east corner of bounding box. Test example: 40.81,-73.41
        southwest: Coordinates of south-west corner of bounding box. Test example: 40.62,-74.73
        intersect: Include areas that intersect with bounding box
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/counties/search/in-bounding-box"
    querystring = {'northEast': northeast, 'southWest': southwest, }
    if intersect:
        querystring['intersect'] = intersect
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def countiessearchinradius(radius: int, longitude: int, latitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Counties boundaries in circle with given radius (in GeoJSON format). For a quick view, copy and paste results: http://geojsonlint.com"
    radius: Radius size in miles. Min value is 1, max value is 50 miles.
        longitude: Longitude of radius center. Test example: -74.72
        latitude: Latitude of radius center. Test example: 40.61
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/counties/search/in-radius"
    querystring = {'radius': radius, 'longitude': longitude, 'latitude': latitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def placessearchinradius(latitude: int, longitude: int, radius: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Places boundaries in circle with given radius (in GeoJSON format). For a quick view, copy and paste results: http://geojsonlint.com"
    latitude: Latitude of radius center. Test example: 40.61
        longitude: Longitude of radius center. Test example: -74.72
        radius: Radius size in miles. Min value is 1, max value is 50 miles.
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/places/search/in-radius"
    querystring = {'latitude': latitude, 'longitude': longitude, 'radius': radius, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def placebynameandstate(name: str, state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get boundaries of Place by name and state (in GeoJSON format).  For a quick view, copy and paste results: http://geojsonlint.com"
    name: Full place name. Test example: Madison
        state: State name abbreviation. Test example: AL
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/places/by-name-and-state"
    querystring = {'name': name, 'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zipcodesbyids(ids: str, properties: str='zip,centroid,aland,awater', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get boundaries of given ZIP Codes (in GeoJSON format). For a quick view, copy and paste results: http://geojsonlint.com"
    ids: Array of comma separated zipCodes/ids. Max size is 200. Test example: 10021,10022,10023
        properties: Customize feature properties to include. In case of ZIP Code - set of [\"zip\", \"centroid\", \"aland\", \"awater\"]. Default value includes all properties: zip,centroid,aland,awater . 
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/zipcodes/by-ids"
    querystring = {'ids': ids, }
    if properties:
        querystring['properties'] = properties
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def placesbyids(ids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get boundaries of given Places (in GeoJSON format). For a quick view, copy and paste results: http://geojsonlint.com"
    ids: Array of comma separated place ids. Test example: 0126896,0162688,0145784
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/places/by-ids"
    querystring = {'ids': ids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def placeslistbystate(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a plain list of Places of a state. Result is a plain list of place ids, not GeoJSON format. To retrieve boundaries of all Places of a state you may query method "placesByIds" with a portion of current method output."
    state: State name abbreviation. Test example: NY
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/places/by-state/list"
    querystring = {'state': state, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def placessearchinboundingbox(northeast: str, southwest: str, intersect: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Places boundaries in provided bounding box (in GeoJSON format). For a quick view, copy and paste results: http://geojsonlint.com"
    northeast: Coordinates of north-east corner of bounding box. Test example: 40.81,-73.41
        southwest: Coordinates of south-west corner of bounding box. Test example: 40.62,-74.73
        intersect: Include areas that intersect with bounding box
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/places/search/in-bounding-box"
    querystring = {'northEast': northeast, 'southWest': southwest, }
    if intersect:
        querystring['intersect'] = intersect
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statessearchinradius(latitude: int, radius: int, longitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get States boundaries in circle with given radius (in GeoJSON format). For a quick view, copy and paste results: http://geojsonlint.com"
    latitude: Latitude of radius center. Test example: 40.61
        radius: Radius size in miles. Min value is 1, max value is 50 miles.
        longitude: Longitude of radius center. Test example: -74.72
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/states/search/in-radius"
    querystring = {'latitude': latitude, 'radius': radius, 'longitude': longitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def statessearchinboundingbox(northeast: str, southwest: str, intersect: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get States boundaries in provided bounding box (in GeoJSON format). For a quick view, copy and paste results: http://geojsonlint.com"
    northeast: Coordinates of north-east corner of bounding box. Test example: 40.81,-73.41
        southwest: Coordinates of south-west corner of bounding box. Test example: 40.62,-74.73
        intersect: Include areas that intersect with bounding box
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/states/search/in-bounding-box"
    querystring = {'northEast': northeast, 'southWest': southwest, }
    if intersect:
        querystring['intersect'] = intersect
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def states(states: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get boundaries of given States (in GeoJSON format). By default, returns boundaries of all States.  For a quick view, copy and paste results: http://geojsonlint.com"
    states: Array of comma separated State name abbreviations. If the parameter is absent, boundaries of all States requested. Test example: NY,CA,DE,AK
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/states"
    querystring = {}
    if states:
        querystring['states'] = states
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def zipcodesbycounty(countyids: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ZIP Codes boundaries of County (in GeoJSON format). For a quick view, copy and paste results: http://geojsonlint.com"
    countyids: Array of  County ids. Test example: 20109
        
    """
    url = f"https://mapreflex.p.rapidapi.com/api/us/v1/zipcodes/by-counties"
    querystring = {'countyIds': countyids, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mapreflex.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

