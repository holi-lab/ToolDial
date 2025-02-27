import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def property_info_basic(property_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get basic attributes for a property."
    
    """
    url = f"https://reit-property-data1.p.rapidapi.com/property-info-basic"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reit-property-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reit_index(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all available data including which REITs are available, the dates of collection, and the property types found in those REITs. No parameters required."
    
    """
    url = f"https://reit-property-data1.p.rapidapi.com/reit-index"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reit-property-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_info_premium(property_id: int, attribute: str='location', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get premium attributes for a property."
    attribute: The premium attribute to query. If no attribute is provided, a list of valid attributes is returned.
        
    """
    url = f"https://reit-property-data1.p.rapidapi.com/property-info-premium"
    querystring = {'property_id': property_id, }
    if attribute:
        querystring['attribute'] = attribute
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reit-property-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feature_search(lon: int, lat: int, distance: int=1000, feature_type: str='starbucks', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for features within distance of a latitude/longitude coordinate pair."
    distance: Distance is in meters. Defaults to 5000.
        
    """
    url = f"https://reit-property-data1.p.rapidapi.com/feature-search"
    querystring = {'lon': lon, 'lat': lat, }
    if distance:
        querystring['distance'] = distance
    if feature_type:
        querystring['feature_type'] = feature_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reit-property-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property(reit: str, geo_level: str='country', geo_value: str='US', qy: str='Q22022', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get IDs of properties. Geography and calendar filters are optional."
    geo_level: Must be used with `geo_value` parameter. Used to specify a geography level to restrict search. Valid `geo_level` values include `zip_code`,`state`,`country`, `countygeoid`, and `msageoid`. Example's of different geo_value values are below.

Note - US states and Canadian provinces are referred to by 2 character codes. All other states and provinces use full name designation

Example usage
- `geo_level = zip_code` and `geo_value = 10010` to get all properties for zip code in NYC
- `geo_level = state` and `geo_value = NY` to get all properties for a REIT in New York state
- `geo_level = country` and `geo_value = US` to get all US properties for a REIT

        geo_value: Must be used with `geo_level` parameter. Used to specify a geography level to restrict search. Valid `geo_level` values include `zip_code`, `state`,`country`,`countygeoid` and `msageoid`.  Examples of different geo_value values are below.

Note - US states and Canadian provinces are referred to by 2 character codes. All other states and provinces use full name designation

Example usage
- `geo_level = zip_code` and `geo_value = 10010` to get all properties for zip code in NYC
- `geo_level = state` and `geo_value = NY` to get all properties for a REIT in New York state
- `geo_level = country` and `geo_value = US` to get all US properties for a REIT

        qy: This is the Quarter & Year (QY) the data was scraped. Use this parameter to query a specifc collection of data. If no QY is provided, defaults to latest data collection for specified REIT.
        
    """
    url = f"https://reit-property-data1.p.rapidapi.com/property"
    querystring = {'reit': reit, }
    if geo_level:
        querystring['geo_level'] = geo_level
    if geo_value:
        querystring['geo_value'] = geo_value
    if qy:
        querystring['qy'] = qy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "reit-property-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

