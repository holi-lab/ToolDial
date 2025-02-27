import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def properties_count(zipcodes: str, bedrooms: str=None, maximumlivingarea: int=None, rooms: str=None, maximumgroundarea: int=None, sortby: int=0, includenewconstructions: bool=None, maximumprice: int=None, transactiontype: int=1, minimumgroundarea: int=None, minimumfloor: int=None, districtids: str=None, minimumlivingarea: int=None, maximumfloor: int=None, realtytypes: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Count total properties returned with options and filters"
    zipcodes: Either zipCodes OR districtIds parameter is required. The postal code, separated by comma for multiple values. Ex : 75,13,...
        bedrooms: Filter by number of bedrooms (1-5). Separated by comma for multiple values. Ex : 2,3
        maximumlivingarea: Filter by maximum living area
        rooms: Filter by number of rooms (1-5). Separated by comma for multiple values. Ex : 2,3
        maximumgroundarea: Filter by maximum ground area
        sortby: One of the following : 0-Pertinence | 1-Prix croissant | 2-Prix decroissant | 10-Du + recent au + ancien | 9-Du + ancien au + recent | 5-Surface croissante | 6-Surface decroissante
        includenewconstructions: Whether or not includes new constructions in listing
        maximumprice: Filter by maximum price
        transactiontype: One of the following : 1-Louer | 2-Acheter | 5-Viager | 6-Investir
        minimumgroundarea: Filter by minimum ground area
        minimumfloor: Filter by minimum number of floors
        districtids: Either zipCodes OR districtIds parameter is required. The value of id fields under 'districts' JSON object returned in .../locations/search endpoint. Separated by comma for multiple values. Ex : 133051,133137,...
        minimumlivingarea: Filter by minimum living area
        maximumfloor: Filter by maximum number of floors
        realtytypes: One of the following : 1-Appartement | 2-Maison et Villa | 2048-Chateau | 128-Loft/Atelier/Surface | 4096-Hotel Particulier | 4-Parking/Box | 8-Terrain | 512-Immeuble | 1024-Batiment | 16-Boutique | 32-Local Commercial | 64-Bureau
        
    """
    url = f"https://seloger.p.rapidapi.com/properties/count"
    querystring = {'zipCodes': zipcodes, }
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if maximumlivingarea:
        querystring['maximumLivingArea'] = maximumlivingarea
    if rooms:
        querystring['rooms'] = rooms
    if maximumgroundarea:
        querystring['maximumGroundArea'] = maximumgroundarea
    if sortby:
        querystring['sortBy'] = sortby
    if includenewconstructions:
        querystring['includeNewConstructions'] = includenewconstructions
    if maximumprice:
        querystring['maximumPrice'] = maximumprice
    if transactiontype:
        querystring['transactionType'] = transactiontype
    if minimumgroundarea:
        querystring['minimumGroundArea'] = minimumgroundarea
    if minimumfloor:
        querystring['minimumFloor'] = minimumfloor
    if districtids:
        querystring['districtIds'] = districtids
    if minimumlivingarea:
        querystring['minimumLivingArea'] = minimumlivingarea
    if maximumfloor:
        querystring['maximumFloor'] = maximumfloor
    if realtytypes:
        querystring['realtyTypes'] = realtytypes
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seloger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_detail(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get property detailed information"
    id: The value of id fields returned in .../properties/list or .../properties/list-in-boundary endpoint.
        
    """
    url = f"https://seloger.p.rapidapi.com/properties/detail"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seloger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list_in_boundary(southeastlongitude: int, southeastlatitude: int, zipcodes: str, northwestlongitude: int, northwestlatitude: int, maximumfloor: int=None, maximumprice: int=None, minimumfloor: int=None, minimumlivingarea: int=None, maximumlivingarea: int=None, bedrooms: str=None, districtids: str=None, rooms: str=None, sortby: int=0, maximumgroundarea: int=None, minimumgroundarea: int=None, includenewconstructions: bool=None, realtytypes: int=1, transactiontype: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties in a bounding box with options and filters"
    southeastlongitude: The south east longitude of bounding box
        southeastlatitude: The south east latitude of bounding box
        zipcodes: Either zipCodes OR districtIds parameter is required. The postal code, separated by comma for multiple values. Ex : 75,13,...
        northwestlongitude: The north west longitude of bounding box
        northwestlatitude: The north west latitude of bounding box
        maximumfloor: Filter by maximum number of floors
        maximumprice: Filter by maximum price
        minimumfloor: Filter by minimum number of floors
        minimumlivingarea: Filter by minimum living area
        maximumlivingarea: Filter by maximum living area
        bedrooms: Filter by number of bedrooms (1-5). Separated by comma for multiple values. Ex : 2,3
        districtids: Either zipCodes OR districtIds parameter is required. The value of id fields under 'districts' JSON object returned in .../locations/search endpoint. Separated by comma for multiple values. Ex : 133051,133137,...
        rooms: Filter by number of rooms (1-5). Separated by comma for multiple values. Ex : 2,3
        sortby: One of the following : 0-Pertinence | 1-Prix croissant | 2-Prix decroissant | 10-Du + recent au + ancien | 9-Du + ancien au + recent | 5-Surface croissante | 6-Surface decroissante
        maximumgroundarea: Filter by maximum ground area
        minimumgroundarea: Filter by minimum ground area
        includenewconstructions: Whether or not includes new constructions in listing
        realtytypes: One of the following : 1-Appartement | 2-Maison et Villa | 2048-Chateau | 128-Loft/Atelier/Surface | 4096-Hotel Particulier | 4-Parking/Box | 8-Terrain | 512-Immeuble | 1024-Batiment | 16-Boutique | 32-Local Commercial | 64-Bureau
        transactiontype: One of the following : 1-Louer | 2-Acheter | 5-Viager | 6-Investir
        
    """
    url = f"https://seloger.p.rapidapi.com/properties/list-in-boundary"
    querystring = {'southEastLongitude': southeastlongitude, 'southEastLatitude': southeastlatitude, 'zipCodes': zipcodes, 'northWestLongitude': northwestlongitude, 'northWestLatitude': northwestlatitude, }
    if maximumfloor:
        querystring['maximumFloor'] = maximumfloor
    if maximumprice:
        querystring['maximumPrice'] = maximumprice
    if minimumfloor:
        querystring['minimumFloor'] = minimumfloor
    if minimumlivingarea:
        querystring['minimumLivingArea'] = minimumlivingarea
    if maximumlivingarea:
        querystring['maximumLivingArea'] = maximumlivingarea
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if districtids:
        querystring['districtIds'] = districtids
    if rooms:
        querystring['rooms'] = rooms
    if sortby:
        querystring['sortBy'] = sortby
    if maximumgroundarea:
        querystring['maximumGroundArea'] = maximumgroundarea
    if minimumgroundarea:
        querystring['minimumGroundArea'] = minimumgroundarea
    if includenewconstructions:
        querystring['includeNewConstructions'] = includenewconstructions
    if realtytypes:
        querystring['realtyTypes'] = realtytypes
    if transactiontype:
        querystring['transactionType'] = transactiontype
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seloger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list(zipcodes: str, pageindex: int=1, pagesize: int=50, bedrooms: str=None, rooms: str=None, minimumlivingarea: int=None, includenewconstructions: bool=None, realtytypes: int=1, maximumgroundarea: int=None, maximumlivingarea: int=None, minimumgroundarea: int=None, minimumfloor: int=None, maximumprice: int=None, districtids: str=None, transactiontype: int=1, sortby: int=0, maximumfloor: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties with options and filters"
    zipcodes: Either zipCodes OR districtIds parameter is required. The postal code, separated by comma for multiple values. Ex : 75,13,...
        pageindex: For paging purpose
        pagesize: For paging purpose (Max is 50)
        bedrooms: Filter by number of bedrooms (1-5). Separated by comma for multiple values. Ex : 2,3
        rooms: Filter by number of rooms (1-5). Separated by comma for multiple values. Ex : 2,3
        minimumlivingarea: Filter by minimum living area
        includenewconstructions: Whether or not includes new constructions in listing
        realtytypes: One of the following : 1-Appartement | 2-Maison et Villa | 2048-Chateau | 128-Loft/Atelier/Surface | 4096-Hotel Particulier | 4-Parking/Box | 8-Terrain | 512-Immeuble | 1024-Batiment | 16-Boutique | 32-Local Commercial | 64-Bureau
        maximumgroundarea: Filter by maximum ground area
        maximumlivingarea: Filter by maximum living area
        minimumgroundarea: Filter by minimum ground area
        minimumfloor: Filter by minimum number of floors
        maximumprice: Filter by maximum price
        districtids: Either zipCodes OR districtIds parameter is required. The value of id fields under 'districts' JSON object returned in .../locations/search endpoint. Separated by comma for multiple values. Ex : 133051,133137,...
        transactiontype: One of the following : 1-Louer | 2-Acheter | 5-Viager | 6-Investir
        sortby: One of the following : 0-Pertinence | 1-Prix croissant | 2-Prix decroissant | 10-Du + recent au + ancien | 9-Du + ancien au + recent | 5-Surface croissante | 6-Surface decroissante
        maximumfloor: Filter by maximum number of floors
        
    """
    url = f"https://seloger.p.rapidapi.com/properties/list"
    querystring = {'zipCodes': zipcodes, }
    if pageindex:
        querystring['pageIndex'] = pageindex
    if pagesize:
        querystring['pageSize'] = pagesize
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if rooms:
        querystring['rooms'] = rooms
    if minimumlivingarea:
        querystring['minimumLivingArea'] = minimumlivingarea
    if includenewconstructions:
        querystring['includeNewConstructions'] = includenewconstructions
    if realtytypes:
        querystring['realtyTypes'] = realtytypes
    if maximumgroundarea:
        querystring['maximumGroundArea'] = maximumgroundarea
    if maximumlivingarea:
        querystring['maximumLivingArea'] = maximumlivingarea
    if minimumgroundarea:
        querystring['minimumGroundArea'] = minimumgroundarea
    if minimumfloor:
        querystring['minimumFloor'] = minimumfloor
    if maximumprice:
        querystring['maximumPrice'] = maximumprice
    if districtids:
        querystring['districtIds'] = districtids
    if transactiontype:
        querystring['transactionType'] = transactiontype
    if sortby:
        querystring['sortBy'] = sortby
    if maximumfloor:
        querystring['maximumFloor'] = maximumfloor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seloger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_search(searchterm: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search location by term or phrase"
    searchterm: Any term or phrase that you are familiar with
        
    """
    url = f"https://seloger.p.rapidapi.com/locations/search"
    querystring = {'searchTerm': searchterm, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "seloger.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

