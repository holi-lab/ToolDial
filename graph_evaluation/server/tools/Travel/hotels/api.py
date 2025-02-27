import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reviews_list_deprecated(is_id: int, loc: str='en_US', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all reviews of specific property"
    id: The value of id field that received from .../properties/list endpoint
        loc: The language code
        page: For paging purpose
        
    """
    url = f"https://hotels4.p.rapidapi.com/reviews/list"
    querystring = {'id': is_id, }
    if loc:
        querystring['loc'] = loc
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_v2_list_deprecated(hotelid: str, revieworder: str='date_newest_first', paginationurl: str=None, triptypefilter: str='all', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all reviews of specific property"
    hotelid: The value of id field that received from …/properties/list endpoint
        revieworder: One of the following : rating&#95;highest&#95;first|rating&#95;lowest&#95;first|date&#95;newest&#95;first
        paginationurl: The value of nextURL field returned right in this endpoint, used to load next page.
        triptypefilter: One of the following : all|romance|family|with-friends|other
        
    """
    url = f"https://hotels4.p.rapidapi.com/reviews/v2/list"
    querystring = {'hotelId': hotelid, }
    if revieworder:
        querystring['reviewOrder'] = revieworder
    if paginationurl:
        querystring['paginationURL'] = paginationurl
    if triptypefilter:
        querystring['tripTypeFilter'] = triptypefilter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_details_deprecated(is_id: int, children6: str=None, children1: str=None, adults3: int=None, adults4: int=None, adults6: int=None, children3: str=None, adults2: int=None, adults7: int=None, currency: str='USD', locale: str='en_US', children7: str=None, children4: str=None, children8: str=None, children5: str=None, children2: str=None, adults8: int=None, adults5: int=None, adults1: int=1, checkout: str='2020-01-15', checkin: str='2020-01-08', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available information of a property"
    id: The value of id field that returned in properties/list endpoint
        children6: The age of every children separated by comma in the sixth room
        children1: The age of every children separated by comma in the first room. Ex : 5,11
        adults3: The number of adults in the third room
        adults4: The number of adults in the fourth room
        adults6: The number of adults in the sixth room
        children3: The age of every children separated by comma in the third room
        adults2: The number of adults in the second room
        adults7: The number of adults in the seventh room
        currency: The currency code
        locale: The language code
        children7: The age of every children separated by comma in the seventh room
        children4: The age of every children separated by comma in the fourth room
        children8: The age of every children separated by comma in the eighth room
        children5: The age of every children separated by comma in the fifth room
        children2: The age of every children separated by comma in the second room
        adults8: The number of adults in the eighth room
        adults5: The number of adults in the fifth room
        adults1: The number of adults in the first room
        checkout: The check-out date at hotel
        checkin: The check-in date at hotel
        
    """
    url = f"https://hotels4.p.rapidapi.com/properties/get-details"
    querystring = {'id': is_id, }
    if children6:
        querystring['children6'] = children6
    if children1:
        querystring['children1'] = children1
    if adults3:
        querystring['adults3'] = adults3
    if adults4:
        querystring['adults4'] = adults4
    if adults6:
        querystring['adults6'] = adults6
    if children3:
        querystring['children3'] = children3
    if adults2:
        querystring['adults2'] = adults2
    if adults7:
        querystring['adults7'] = adults7
    if currency:
        querystring['currency'] = currency
    if locale:
        querystring['locale'] = locale
    if children7:
        querystring['children7'] = children7
    if children4:
        querystring['children4'] = children4
    if children8:
        querystring['children8'] = children8
    if children5:
        querystring['children5'] = children5
    if children2:
        querystring['children2'] = children2
    if adults8:
        querystring['adults8'] = adults8
    if adults5:
        querystring['adults5'] = adults5
    if adults1:
        querystring['adults1'] = adults1
    if checkout:
        querystring['checkOut'] = checkout
    if checkin:
        querystring['checkIn'] = checkin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list_deprecated(checkout: str, destinationid: int, pagenumber: int, checkin: str, adults1: int, pagesize: int, accommodationids: str=None, guestratingmin: int=None, children4: str=None, children8: str=None, locale: str='en_US', pricemin: int=None, pricemax: int=None, adults8: int=None, adults6: int=None, adults4: int=None, starratings: str=None, sortorder: str='PRICE', children2: str=None, adults2: int=None, adults3: int=None, adults5: int=None, currency: str='USD', amenityids: str=None, children6: str=None, children5: str=None, children1: str=None, children3: str=None, children7: str=None, landmarkids: str=None, themeids: str=None, adults7: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties with options and filters"
    checkout: The check-out date at hotel, formated as yyyy-MM-dd
        destinationid: The value of destinationId returned in locations/search endpoint
        pagenumber: The page number in which data is display
        checkin: The check-in date at hotel, formated as yyyy-MM-dd
        adults1: The number of adults in first room
        pagesize: Total items returned in every requests (max 25)
        accommodationids: Check for suitable value (separated by comma to specify multiple values) in accommodationType field returned right in this endpoint
        guestratingmin: Check for suitable min rating in filters/guestRating/range/min field returned right in this endpoint. Please use integer value.
        children4: The age of every children separated by comma in the fourth room
        children8: The age of every children separated by comma in the eighth room
        locale: The language code, get all supported language code from .../get-meta-data endpoint
        pricemin: Check for suitable min price in filters/price/range/min field returned right in this endpoint
        pricemax: Check for suitable max price in filters/price/range/max field returned right in this endpoint
        adults8: The number of adults in the eighth room
        adults6: The number of adults in the sixth room
        adults4: The number of adults in the fourth room
        starratings: Check for suitable value (separated by comma to specify multiple values) in starRating returned right in this endpoint. Ex : 2,3
        sortorder: One of the following is allowed BEST&#95;SELLER|STAR&#95;RATING&#95;HIGHEST&#95;FIRST|STAR&#95;RATING&#95;LOWEST&#95;FIRST|DISTANCE&#95;FROM&#95;LANDMARK|GUEST&#95;RATING|PRICE&#95;HIGHEST&#95;FIRST|PRICE
        children2: The age of every children separated by comma in the second room
        adults2: The number of adults in the second room
        adults3: The number of adults in the third room
        adults5: The number of adults in the fifth room
        currency: The currency code
        amenityids: Check for suitable value (separated by comma to specify multiple values) in facilities field returned right in this endpoint
        children6: The age of every children separated by comma in the sixth room
        children5: The age of every children separated by comma in the fifth room
        children1: The age of every children separated by comma in the first room. Ex : 5,11
        children3: The age of every children separated by comma in the third room
        children7: The age of every children separated by comma in the seventh room
        landmarkids: Check for suitable value (separated by comma to specify multiple values) in landmarks field returned right in this endpoint
        themeids: Check for suitable value (separated by comma to specify multiple values) in themesAndTypes field returned right in this endpoint
        adults7: The number of adults in the seventh room
        
    """
    url = f"https://hotels4.p.rapidapi.com/properties/list"
    querystring = {'checkOut': checkout, 'destinationId': destinationid, 'pageNumber': pagenumber, 'checkIn': checkin, 'adults1': adults1, 'pageSize': pagesize, }
    if accommodationids:
        querystring['accommodationIds'] = accommodationids
    if guestratingmin:
        querystring['guestRatingMin'] = guestratingmin
    if children4:
        querystring['children4'] = children4
    if children8:
        querystring['children8'] = children8
    if locale:
        querystring['locale'] = locale
    if pricemin:
        querystring['priceMin'] = pricemin
    if pricemax:
        querystring['priceMax'] = pricemax
    if adults8:
        querystring['adults8'] = adults8
    if adults6:
        querystring['adults6'] = adults6
    if adults4:
        querystring['adults4'] = adults4
    if starratings:
        querystring['starRatings'] = starratings
    if sortorder:
        querystring['sortOrder'] = sortorder
    if children2:
        querystring['children2'] = children2
    if adults2:
        querystring['adults2'] = adults2
    if adults3:
        querystring['adults3'] = adults3
    if adults5:
        querystring['adults5'] = adults5
    if currency:
        querystring['currency'] = currency
    if amenityids:
        querystring['amenityIds'] = amenityids
    if children6:
        querystring['children6'] = children6
    if children5:
        querystring['children5'] = children5
    if children1:
        querystring['children1'] = children1
    if children3:
        querystring['children3'] = children3
    if children7:
        querystring['children7'] = children7
    if landmarkids:
        querystring['landmarkIds'] = landmarkids
    if themeids:
        querystring['themeIds'] = themeids
    if adults7:
        querystring['adults7'] = adults7
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_meta_data_deprecated(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get locale meta data"
    
    """
    url = f"https://hotels4.p.rapidapi.com/get-meta-data"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_v2_search_deprecated(query: str, locale: str='en_US', currency: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for related locations and suggestions"
    query: Name of countries, cities, districts, places, etc...
        locale: The language code
        currency: The currency code
        
    """
    url = f"https://hotels4.p.rapidapi.com/locations/v2/search"
    querystring = {'query': query, }
    if locale:
        querystring['locale'] = locale
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_hotel_photos_deprecated(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available photos of property"
    id: The value of id field that received from .../properties/list endpoint
        
    """
    url = f"https://hotels4.p.rapidapi.com/properties/get-hotel-photos"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_hotel_photos_deprecated(is_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available photos of property"
    id: The value of id field that received from .../properties/list endpoint
        
    """
    url = f"https://hotels4.p.rapidapi.com/properties/get-hotel-photos"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_v2_resolve_url(is_id: str, siteid: int=300000001, locale: str='en_US', langid: int=1033, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hotels system is being merged with Expedia. This endpoint helps to convert the old hotel ID (embedded in the shared link, Ex : "ho133528" in hotels.com/ho133528/) to the new one (423818)."
    id: The id (started with ho...) extracted from the shared link. Ex : 'ho546370' from https://www.hoteles.com/ho546370
        siteid: The value of siteId field returned in …/v2/get-meta-data endpoint
        locale: The language code
        langid: The value of languageIdentifier field returned in …/v2/get-meta-data endpoint
        
    """
    url = f"https://hotels4.p.rapidapi.com/properties/v2/resolve-url"
    querystring = {'id': is_id, }
    if siteid:
        querystring['siteid'] = siteid
    if locale:
        querystring['locale'] = locale
    if langid:
        querystring['langid'] = langid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_get_meta_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get locale meta data"
    
    """
    url = f"https://hotels4.p.rapidapi.com/v2/get-meta-data"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_search_deprecated(query: str, locale: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for related locations and suggestions"
    query: Name of countries, cities, districts, places, etc...
        locale: The language code
        
    """
    url = f"https://hotels4.p.rapidapi.com/locations/search"
    querystring = {'query': query, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_v3_search(q: str, langid: int=1033, siteid: int=300000001, locale: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for related locations and suggestions"
    q: Name of countries, cities, districts, places, etc...
        langid: The value of languageIdentifier field returned in .../v2/get-meta-data endpoint
        siteid: The value of siteId field returned in .../v2/get-meta-data endpoint
        locale: The language code
        
    """
    url = f"https://hotels4.p.rapidapi.com/locations/v3/search"
    querystring = {'q': q, }
    if langid:
        querystring['langid'] = langid
    if siteid:
        querystring['siteid'] = siteid
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

