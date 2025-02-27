import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reverse_geocoding(latlng: str, location_type: str=None, result_type: str=None, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Address Lookup"
    latlng: The latitude and longitude values specifying the location for which you wish to obtain the closest, human-readable address.
        location_type: A filter of one or more location types, separated by a pipe (|). If the parameter contains multiple location types, the API returns all addresses that match any of the types. A note about processing: The location_type parameter does not restrict the search to the specified location type(s). Rather, the location_type acts as a post-search filter: the API fetches all results for the specified latlng, then discards those results that do not match the specified location type(s). Note: This parameter is available only for requests that include an API key or a client ID. A list of supported types can be found in the API Overview.
        result_type: A filter of one or more address types, separated by a pipe (|). If the parameter contains multiple address types, the API returns all addresses that match any of the types. A note about processing: The result_type parameter does not restrict the search to the specified address type(s). Rather, the result_type acts as a post-search filter: the API fetches all results for the specified latlng, then discards those results that do not match the specified address type(s). Note: This parameter is available only for requests that include an API key or a client ID. List of all supported types can be found in API Overview
        language: The language in which to return results. See list of all supported languages in API Overvier.
        
    """
    url = f"https://google-maps-geocoding.p.rapidapi.com/geocode/json"
    querystring = {'latlng': latlng, }
    if location_type:
        querystring['location_type'] = location_type
    if result_type:
        querystring['result_type'] = result_type
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-maps-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def geocoding(address: str, components: str=None, language: str='en', bounds: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Latitude/Longitude Lookup"
    address: The street address that you want to geocode, in the format used by the national postal service of the country concerned. Additional address elements such as business names and unit, suite or floor numbers should be avoided.
        components: A components filter with elements separated by a pipe (|). The components filter is also accepted as an optional parameter if an address is provided. Each element in the components filter consists of a component:value pair, and fully restricts the results from the geocoder. See more information about component filtering in the API Overview tab.
        language: The language in which to return results. See the list of supported languages in API Overview tab.
        bounds: bounds — The bounding box of the viewport within which to bias geocode results more prominently. This parameter will only influence, not fully restrict, results from the geocoder. (For more information see Viewport Biasing in the API Overview.)
        
    """
    url = f"https://google-maps-geocoding.p.rapidapi.com/geocode/json"
    querystring = {'address': address, }
    if components:
        querystring['components'] = components
    if language:
        querystring['language'] = language
    if bounds:
        querystring['bounds'] = bounds
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "google-maps-geocoding.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

