import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def limitbyviewbox(bounded: int=1, key: str='{{Your_API_Key}}', viewbox: str='-132.84908,47.69382,-70.44674,30.82531', q: str='Empire State', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Specify the preferred area to find search results.
		
		> The Autocomplete API is a variant of the Search API that returns place predictions in response to an HTTP request. The request specifies a textual search string and optional geographic bounds. The service can be used to provide autocomplete functionality for text-based geographic searches, by returning places such as businesses, addresses and points of interest as a user types.
		
		> The Autocomplete API can match on full words as well as substrings. Applications can therefore send queries as the user types, to provide on-the-fly place predictions."
    bounded: Restrict result to items contained within the bounds specified in the viewbox parameter [0] [1]
        key: Your API key
        viewbox: The preferred area to find search results. Any two corner points of the box - max_lon,max_lat,min_lon,min_lat or min_lon,min_lat,max_lon,max_lat - are accepted in any order as long as they span a real box. Currently, this option in the Autocomplete API only increases weigtage of results inside the viewbox. It does not restrict results to this box.
        q: Query string to search for
        
    """
    url = f"https://locationiq2.p.rapidapi.com/v1/autocomplete"
    querystring = {}
    if bounded:
        querystring['bounded'] = bounded
    if key:
        querystring['key'] = key
    if viewbox:
        querystring['viewbox'] = viewbox
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "locationiq2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listofallnearbypoisexceptgyms(lat: int=40, key: str='{{Your_API_Key}}', tag: str='!amenity:gym', lon: int=-73, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Nearby Points of Interest (PoI) API returns specified PoIs or Places around a given coordinate.
		
		"
    lat: Latitude of the location to generate the PoI list for.
        key: Authentication Key
        tag: PoI to generate the list for. Defaults to country (check Nearby-Countries). 
        lon: Longitude of the location to generate the PoI list for.
        
    """
    url = f"https://locationiq2.p.rapidapi.com/v1/nearby"
    querystring = {}
    if lat:
        querystring['lat'] = lat
    if key:
        querystring['key'] = key
    if tag:
        querystring['tag'] = tag
    if lon:
        querystring['lon'] = lon
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "locationiq2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generalusage(lat: int=17, key: str='{{Your_API_Key}}', lon: int=78, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Timezone API provides time offset data for locations on the surface of the earth."
    lat: Latitude of the location
        key: Authentication key
        lon: Longitude of the location
        
    """
    url = f"https://locationiq2.p.rapidapi.com/v1/timezone"
    querystring = {}
    if lat:
        querystring['lat'] = lat
    if key:
        querystring['key'] = key
    if lon:
        querystring['lon'] = lon
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "locationiq2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generalusage(key: str='{{Your_API_Key}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    key: Your API Access Token
        
    """
    url = f"https://locationiq2.p.rapidapi.com/v3/staticmap"
    querystring = {}
    if key:
        querystring['key'] = key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "locationiq2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def postalcodequery(format: str='json', key: str='{{Your_API_Key}}', countrycodes: str='us', postalcode: int=10001, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It takes postalcode & countrycode as input.
		
		> The Search API allows converting addresses, such as a street address, into geographic coordinates (latitude and longitude). These coordinates can serve various use-cases, from placing markers on a map to helping algorithms determine nearby bus stops. This process is also known as Forward Geocoding."
    format: Output Format [json | xml]
        key: Your API Key
        countrycodes: Limit search to a list of countries. 
        postalcode: Alternative query string format for postal code requests that uses a special postal code dataset for geocoding. Do not combine with q=<query> or other structured parameters for postal code search. Combine with countrycodes=<countrycode> parameter for a better response
        
    """
    url = f"https://locationiq2.p.rapidapi.com/v1/search"
    querystring = {}
    if format:
        querystring['format'] = format
    if key:
        querystring['key'] = key
    if countrycodes:
        querystring['countrycodes'] = countrycodes
    if postalcode:
        querystring['postalcode'] = postalcode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "locationiq2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fixedsourcedestination(service: str, profile: str, coordinates: str, source: str='first', key: str='{{Your_API_Key}}', destination: str='last', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Round trip in London with four stops, starting at the first stop, ending at the last."
    service: One of the following values: directions, matching, matrix, nearest, optimize
        profile: Mode of transportation. Only driving is supported at the moment.
        coordinates: String of format {longitude},{latitude};{longitude},{latitude}[;{longitude},{latitude} ...] or polyline({polyline}) or polyline6({polyline6}).
polyline follows Google's polyline format with precision 5
        source: Returned route starts at any or first coordinate [ any | first ]
        destination: Returned route ends at any or last coordinate [ any | last ]
        
    """
    url = f"https://locationiq2.p.rapidapi.com/v1/{service}/{profile}/{coordinates}"
    querystring = {}
    if source:
        querystring['source'] = source
    if key:
        querystring['key'] = key
    if destination:
        querystring['destination'] = destination
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "locationiq2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generalusage(key: str='{{Your_API_Key}}', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The Balance API provides a count of request credits left in the user's account for the day. Balance is reset at midnight UTC everyday (00:00 UTC)."
    
    """
    url = f"https://locationiq2.p.rapidapi.com/v1/balance"
    querystring = {}
    if key:
        querystring['key'] = key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "locationiq2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def usingosmidtype(key: str='{{Your_API_Key}}', osm_type: str='W', osm_id: int=34633854, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It takes OSM ID & Type as input.
		
		> Reverse geocoding with OSM ID & Type is the process of converting an object represented by OSM ID and Type ( node, way and relations [N,W,R] ) to a readable address or place name. This permits the identification of nearby street addresses, places, and/or area subdivisions such as a neighborhood, county, state, or country."
    key: Authentication key.
        osm_type: A specific osm type, node / way / relation to search an address for [ N, W , R ], only applicable when neither (lat lon) is present. That is use in combination osm_id and osm_type or lat and lon to get result
        osm_id: A specific osm node / way / relation to return an address for
        format: Output Format. Defaults to xml
        
    """
    url = f"https://locationiq2.p.rapidapi.com/v1/reverse"
    querystring = {}
    if key:
        querystring['key'] = key
    if osm_type:
        querystring['osm_type'] = osm_type
    if osm_id:
        querystring['osm_id'] = osm_id
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "locationiq2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

