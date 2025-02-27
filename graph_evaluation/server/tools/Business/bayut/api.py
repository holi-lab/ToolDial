import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def properties_detail(externalid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detail information of a property"
    externalid: The value of externalID returned in ..../properties/list endpoint
        
    """
    url = f"https://bayut.p.rapidapi.com/properties/detail"
    querystring = {'externalID': externalid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bayut.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agencies_list(page: int=0, lang: str='en', hitsperpage: int=25, query: str='patriot', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List agencies or search for agencies by name"
    page: For paging purpose
        lang: One of the following : en|ar
        hitsperpage: For paging purpose
        query: Any term or phrase that you are familiar with
        
    """
    url = f"https://bayut.p.rapidapi.com/agencies/list"
    querystring = {}
    if page:
        querystring['page'] = page
    if lang:
        querystring['lang'] = lang
    if hitsperpage:
        querystring['hitsPerPage'] = hitsperpage
    if query:
        querystring['query'] = query
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bayut.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def agencies_get_listings(agencyslug: str, categoryslug: str='commerical-properties', sort: str='price-asc', completionstatus: str=None, rentfrequency: str=None, purpose: str=None, page: int=0, hitsperpage: int=25, lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get listing belonging to an agency"
    agencyslug: The value of hits -> slug field returned in .../agencies/list endpoint
        categoryslug: One of the following : commerical-properties|mixed-use-land|factories|commercial-floors|bulk-units|labour-camps|shops|showrooms|industrial-land|commercial-buildings|commercial-plots|commercial-villas|warehouses|offices|residential-floors|hotel-apartments|penthouse|villas|residential-building|residential-plots|villa-compound|townhouses|apartments
        sort: One of the following : price-desc|price-asc|city-level-score|date-desc|date-asc|verified-score
        completionstatus: One of the following : completed|under-construction
This parameter only takes effect if purpose is for-sale
        rentfrequency: One of the following : monthly|yearly|weekly|daily
The parameter only takes effect if purpose is for-rent
        purpose: One of the following : for-rent|for-sale
        page: For paging purpose
        hitsperpage: For paging purpose
        lang: One of the following : en|ar
        
    """
    url = f"https://bayut.p.rapidapi.com/agencies/get-listings"
    querystring = {'agencySlug': agencyslug, }
    if categoryslug:
        querystring['categorySlug'] = categoryslug
    if sort:
        querystring['sort'] = sort
    if completionstatus:
        querystring['completionStatus'] = completionstatus
    if rentfrequency:
        querystring['rentFrequency'] = rentfrequency
    if purpose:
        querystring['purpose'] = purpose
    if page:
        querystring['page'] = page
    if hitsperpage:
        querystring['hitsPerPage'] = hitsperpage
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bayut.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list(locationexternalids: str, hasfloorplan: bool=None, areamin: int=None, purpose: str='for-rent', hasvideo: bool=None, haspanorama: bool=None, furnishingstatus: str=None, bathsmin: int=None, sort: str='city-level-score', bathsmax: int=None, page: int=0, agencyexternalids: str=None, hitsperpage: int=25, roomsmin: int=None, roomsmax: int=None, categoryexternalid: int=4, rentfrequency: str='monthly', lang: str='en', pricemin: int=None, pricemax: int=None, areamax: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties with options and filters"
    locationexternalids: The location/externalID fields returned in .../auto-complete endpoint. Separated by comma for multiple locations. Ex : 5002,6020
        purpose: One of the following : for-rent|for-sale
        furnishingstatus: One of the following : unfurnished|furnished
        sort: One of the following : price-desc|price-asc|city-level-score|date-desc|verified-score
        page: For paging purpose
        agencyexternalids: The externalID fields returned in .../agencies/list endpoint. Separated by comma for multiple agencies. Ex : 7737,5002
        hitsperpage: For paging purpose
        categoryexternalid: Apartment -> 4|Townhouses -> 16|Villas -> 3|Penthouses -> 18|Hotel Apartments -> 21|Villa Compound -> 19|Residential Plot -> 14|Residential Floor -> 12|Residential Building -> 17|Office -> 5|Shop -> 6|Warehouse -> 7|Labour camp -> 9|Commercial Villa -> 25|Bulk Units -> 20|Commercial Plot -> 15|Commercial Floor -> 13|Commercial Building -> 10|Factory -> 8|Industrial Land -> 22|Mixed Use Land -> 23|Showroom -> 24|Other Commercial -> 11
        rentfrequency: One of the following : monthly|yearly|weekly|daily
        lang: One of the following : en|ar
        
    """
    url = f"https://bayut.p.rapidapi.com/properties/list"
    querystring = {'locationExternalIDs': locationexternalids, }
    if hasfloorplan:
        querystring['hasFloorPlan'] = hasfloorplan
    if areamin:
        querystring['areaMin'] = areamin
    if purpose:
        querystring['purpose'] = purpose
    if hasvideo:
        querystring['hasVideo'] = hasvideo
    if haspanorama:
        querystring['hasPanorama'] = haspanorama
    if furnishingstatus:
        querystring['furnishingStatus'] = furnishingstatus
    if bathsmin:
        querystring['bathsMin'] = bathsmin
    if sort:
        querystring['sort'] = sort
    if bathsmax:
        querystring['bathsMax'] = bathsmax
    if page:
        querystring['page'] = page
    if agencyexternalids:
        querystring['agencyExternalIDs'] = agencyexternalids
    if hitsperpage:
        querystring['hitsPerPage'] = hitsperpage
    if roomsmin:
        querystring['roomsMin'] = roomsmin
    if roomsmax:
        querystring['roomsMax'] = roomsmax
    if categoryexternalid:
        querystring['categoryExternalID'] = categoryexternalid
    if rentfrequency:
        querystring['rentFrequency'] = rentfrequency
    if lang:
        querystring['lang'] = lang
    if pricemin:
        querystring['priceMin'] = pricemin
    if pricemax:
        querystring['priceMax'] = pricemax
    if areamax:
        querystring['areaMax'] = areamax
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bayut.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(query: str, hitsperpage: int=25, lang: str='en', page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestions of locations by term or phrase"
    query: Any term or phrase that you are familiar with
        hitsperpage: For paging purpose
        lang: One of the following : en|ar
        page: For paging purpose
        
    """
    url = f"https://bayut.p.rapidapi.com/auto-complete"
    querystring = {'query': query, }
    if hitsperpage:
        querystring['hitsPerPage'] = hitsperpage
    if lang:
        querystring['lang'] = lang
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "bayut.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

