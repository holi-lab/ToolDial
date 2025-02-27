import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def schools_list(lat: int, lon: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List schools around a GEO location"
    lat: The latitude of GEO location
        lon: The longitude of GEO location
        
    """
    url = f"https://realty-in-au.p.rapidapi.com/schools/list"
    querystring = {'lat': lat, 'lon': lon, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-au.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list(searchlocation: str, channel: str, searchlocationsubtext: str, type: str, keywords: str=None, minimumbedrooms: int=None, minimumcars: int=None, maximumbedrooms: int=None, maximumprice: int=None, surroundingsuburbs: bool=None, minimumprice: int=None, page: int=1, propertytypes: str=None, constructionstatus: str=None, minimumbathroom: int=None, sorttype: str='relevance', ex_under_contract: bool=None, pagesize: int=30, minimumlandsize: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties with options and filters"
    searchlocation: The value of text field returned in .../auto-complete endpoint
        channel: One of the following : buy|rent|sold
        searchlocationsubtext: The value of subtext field returned in .../auto-complete endpoint
        type: The value of region field returned in .../auto-complete endpoint
        keywords: Facilities you are looking for. Separated by comma for multiple options. Ex : pool,garage,etc...
        page: For paging purpose
        propertytypes: Ignore or one of the following : townhouse|unit apartment|retire|acreage|land|unitblock|house|villa|rural. Separated by comma for multiple options. Ex : townhouse,house,villa
        constructionstatus: Ignore or one of the following : established|new
        sorttype: One of the following relevance|new-asc|new-desc|price-asc|price-desc
        pagesize: The number of items returned per response. For paging purpose (max is 30)
        minimumlandsize: In m2
        
    """
    url = f"https://realty-in-au.p.rapidapi.com/properties/list"
    querystring = {'searchLocation': searchlocation, 'channel': channel, 'searchLocationSubtext': searchlocationsubtext, 'type': type, }
    if keywords:
        querystring['keywords'] = keywords
    if minimumbedrooms:
        querystring['minimumBedrooms'] = minimumbedrooms
    if minimumcars:
        querystring['minimumCars'] = minimumcars
    if maximumbedrooms:
        querystring['maximumBedrooms'] = maximumbedrooms
    if maximumprice:
        querystring['maximumPrice'] = maximumprice
    if surroundingsuburbs:
        querystring['surroundingSuburbs'] = surroundingsuburbs
    if minimumprice:
        querystring['minimumPrice'] = minimumprice
    if page:
        querystring['page'] = page
    if propertytypes:
        querystring['propertyTypes'] = propertytypes
    if constructionstatus:
        querystring['constructionStatus'] = constructionstatus
    if minimumbathroom:
        querystring['minimumBathroom'] = minimumbathroom
    if sorttype:
        querystring['sortType'] = sorttype
    if ex_under_contract:
        querystring['ex-under-contract'] = ex_under_contract
    if pagesize:
        querystring['pageSize'] = pagesize
    if minimumlandsize:
        querystring['minimumLandSize'] = minimumlandsize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-au.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_detail(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information of a property"
    id: The value of 'listingId' field returned in .../properties/list endpoint or 'id' field returned in .../auto-complete endpoint with type as listing
        
    """
    url = f"https://realty-in-au.p.rapidapi.com/properties/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-au.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_lookup_deprecating(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Look for off-market property by id
		* Use .../properties/v2/lookup endpoint instead for more data returned."
    id: The value of 'id' field returned in .../auto-complete endpoint with display as 'Property history'
        
    """
    url = f"https://realty-in-au.p.rapidapi.com/properties/lookup"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-au.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_v2_lookup(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Look for off-market property by id"
    id: The value of 'id' field returned in .../auto-complete endpoint with display as 'Property history'
        
    """
    url = f"https://realty-in-au.p.rapidapi.com/properties/v2/lookup"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-au.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestions by term or phrase. The returned data will be used with .../properties/list endpoint"
    query: Any term or phrase you are familiar with. It can be an address to get the property id directly to use with .../properties/detail endpoint.
        
    """
    url = f"https://realty-in-au.p.rapidapi.com/auto-complete"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "realty-in-au.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

