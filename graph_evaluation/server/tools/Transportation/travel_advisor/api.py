import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def flights_create_session_deprecating(d1: str, dd1: str, o1: str, tc: str=None, ts: int=None, ta: int=1, currency: str='USD', d2: str=None, o2: str=None, c: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "* TripAdvisor stops providing "Flights feature" via their mobile application, the web version is going to be down in the future.
		Create new session for searching flights tickets of all airlines around the world. The flight APIs need to use as following : Firstly, you init a search session by using the create-session endpoint and get sid value Secondly, you repeatedly call poll endpoint until the summary/c field is true to get final total flight by summary/nr value. o parameter must be 0 otherwise the summary field is not returned. Thirdly, you repeatedly call poll endpoint with o increasing by n value step by step until you reach total flight."
    d1: The airport code of first destination location
        dd1: The date of departure from first origin, the format is yyyy-MM-dd. Ex : 2020-05-15
        o1: Airport code of first origin
        tc: The age of every children separated by comma. Ex : 11,5
        ts: The number of seniors
        ta: The number of adults
        currency: The currency code
        d2: The airport code of second destination location
        o2: The airport code of second origin location
        c: Cabin code, such as 0 - Economy | 1- Business Class | 2 - First Class | 3 - Premium Economy
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/flights/create-session"
    querystring = {'d1': d1, 'dd1': dd1, 'o1': o1, }
    if tc:
        querystring['tc'] = tc
    if ts:
        querystring['ts'] = ts
    if ta:
        querystring['ta'] = ta
    if currency:
        querystring['currency'] = currency
    if d2:
        querystring['d2'] = d2
    if o2:
        querystring['o2'] = o2
    if c:
        querystring['c'] = c
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def airports_search_deprecating(query: str, locale: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "* TripAdvisor stops providing "Flights feature" via their mobile application, the web version is going to be down in the future.
		Search for airport code by countries', cities', etc... name"
    query: Name of cities, districts, places, etc...
        locale: The language code
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/airports/search"
    querystring = {'query': query, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def attractions_list_deprecated(location_id: int, lunit: str='km', offset: int=None, subcategory: int=None, currency: str='USD', bookable_first: bool=None, limit: int=None, min_rating: int=None, lang: str='en_US', sort: str='recommended', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List attractions of specific location by location_id"
    location_id: The value of location_id field that returned in locations/search endpoint
        lunit: One of the followings km|mi
        offset: The number of items to ignore for paging purpose
        subcategory: Attraction category - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (only one value is allowed at a time) returned right in this endpoint
        currency: The currency code
        bookable_first: Book online first
        limit: The number of items per response (max 30)
        min_rating: Rating - Min 3 max 5
        lang: The language code
        sort: One of following recommended|ranking
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/attractions/list"
    querystring = {'location_id': location_id, }
    if lunit:
        querystring['lunit'] = lunit
    if offset:
        querystring['offset'] = offset
    if subcategory:
        querystring['subcategory'] = subcategory
    if currency:
        querystring['currency'] = currency
    if bookable_first:
        querystring['bookable_first'] = bookable_first
    if limit:
        querystring['limit'] = limit
    if min_rating:
        querystring['min_rating'] = min_rating
    if lang:
        querystring['lang'] = lang
    if sort:
        querystring['sort'] = sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_filters_list_deprecated(location_id: int, adults: int=1, rooms: int=1, lang: str='en_US', pricesmin: int=None, subcategory: str=None, currency: str='USD', zff: str=None, nights: int=2, amenities: str=None, checkin: str=None, hotel_class: str=None, child_rm_ages: str=None, sort: str='recommended', order: str='asc', pricesmax: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available options and filters"
    location_id: The value of location_id field that returned in locations/search endpoint
        adults: The number of adults in all rooms
        rooms: The number of rooms
        lang: The language code
        pricesmin: Check for suitable price range in filters/prices_slider field returned in hotel-filters/list endpoint. For exmaple : $120 -> 120
        subcategory: Check for suitable value of filters/subcategory field (separated by comma to filter by multiple values. Ex : hotel,bb,specialty) returned in hotel-filters/list endpoint
        currency: The currency code
        zff: Hotel Style - Check for suitable value of filters/zff field (separated by comma to filter by multiple values. Ex : 4,6) returned in hotel-filters/list endpoint
        nights: The number of nights to live
        amenities: Check for suitable value of filters/amenities field (separated by comma to filter by multiple values. Ex : beach,bar&#95;lounge,airport&#95;transportation) returned in hotel-filters/list endpoint
        checkin: The check-in date at hotel, the format is yyyy-MM-dd. Ex : 2020-05-15
        hotel_class: Check for suitable value of filters/hotel_class field (separated by comma to filter by multiple values. Ex : 1,2,3) returned in hotel-filters/list endpoint
        child_rm_ages: The age of every children separated by comma in all rooms. Ex : 7,10
        sort: One of the followings : recommended|popularity|price
        order: One of the followings : asc|desc, this param is used with sort by price
        pricesmax: Check for suitable price range in filters/prices_slider field returned in hotel-filters/list endpoint. For exmaple : $120 -> 120
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/hotel-filters/list"
    querystring = {'location_id': location_id, }
    if adults:
        querystring['adults'] = adults
    if rooms:
        querystring['rooms'] = rooms
    if lang:
        querystring['lang'] = lang
    if pricesmin:
        querystring['pricesmin'] = pricesmin
    if subcategory:
        querystring['subcategory'] = subcategory
    if currency:
        querystring['currency'] = currency
    if zff:
        querystring['zff'] = zff
    if nights:
        querystring['nights'] = nights
    if amenities:
        querystring['amenities'] = amenities
    if checkin:
        querystring['checkin'] = checkin
    if hotel_class:
        querystring['hotel_class'] = hotel_class
    if child_rm_ages:
        querystring['child_rm_ages'] = child_rm_ages
    if sort:
        querystring['sort'] = sort
    if order:
        querystring['order'] = order
    if pricesmax:
        querystring['pricesmax'] = pricesmax
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def answers_list_deprecated(question_id: int, offset: int=None, limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List answers related to a questions by its id"
    question_id: The value of id field that returned in questions/list endpoint
        offset: The number of items to ignore for paging purpose
        limit: The number of items per response (max 10)
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/answers/list"
    querystring = {'question_id': question_id, }
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def photos_list_deprecated(location_id: int, currency: str='USD', limit: int=50, lang: str='en_US', offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List photos related to a location by its id"
    location_id: The value of location_id field that returned in hotels/list, restaurants/list, or attractions/list endpoints
        currency: The currency code
        limit: The number of items per response (max 50)
        lang: The language code
        offset: The number of items to ignore for paging purpose
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/photos/list"
    querystring = {'location_id': location_id, }
    if currency:
        querystring['currency'] = currency
    if limit:
        querystring['limit'] = limit
    if lang:
        querystring['lang'] = lang
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tips_list(location_id: int, offset: int=None, lang: str='en_US', currency: str='USD', limit: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List tips of specific hotel by its location_id"
    location_id: The value of location_id field that returned in hotels/list endpoint
        offset: The number of items to ignore for paging purpose
        currency: The currency code
        limit: The number of items per response (max 20)
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/tips/list"
    querystring = {'location_id': location_id, }
    if offset:
        querystring['offset'] = offset
    if lang:
        querystring['lang'] = lang
    if currency:
        querystring['currency'] = currency
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keywords_list(location_id: int, limit: int=10, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List interesting keywords related to a specific location by its location_id"
    location_id: The value of location_id field that returned in hotels/list, restaurants/list, or attractions/list endpoints
        limit: The number of items per response (max 10)
        offset: The number of items to ignore for paging purpose
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/keywords/list"
    querystring = {'location_id': location_id, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def questions_list_deprecated(location_id: int, limit: int=10, offset: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List questions related to a location by its id"
    location_id: The value of location_id field that returned in hotels/list, restaurants/list, or attractions/list endpoints
        limit: The number of items per response (max 10)
        offset: The number of items to ignore for paging purpose
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/questions/list"
    querystring = {'location_id': location_id, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotels_list_in_boundary_deprecated(tr_latitude: int, tr_longitude: int, bl_longitude: int, bl_latitude: int, pricesmax: int=None, amenities: str=None, pricesmin: int=None, offset: int=None, rooms: int=None, subcategory: str='hotel,bb,specialty', currency: str='USD', adults: int=1, child_rm_ages: str=None, zff: str=None, limit: int=30, hotel_class: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List hotels by specifying coordinates of bottom left and top right of boundary"
    tr_latitude: Latitude of top right coordinate
        tr_longitude: Longitude of top right coordinate
        bl_longitude: Longitude of bottom left coordinate
        bl_latitude: Latitude of bottom left coordinate
        amenities: Check for suitable value of filters/amenities field (separated by comma to specify multiple values) returned in hotel-filters/list endpoint
        pricesmin: Check for suitable price range in filters/prices_slider field returned in hotel-filters/list endpoint. For exmaple : $10 -> 10
        offset: The number of items to ignore for paging purpose
        rooms: The number of rooms
        subcategory: Check for suitable value of filters/subcategory field (separated by comma to specify multiple values) returned in hotel-filters/list endpoint
        currency: The currency code
        adults: The number of adults in all rooms
        child_rm_ages: The age of every children separated by comma in all rooms
        zff: Hotel Style - Check for suitable value of \\\"filters/zff\\\" field (separated by comma to specify multiple values) returned in hotel-filters/list endpoint
        limit: The number of items per response (max 30)
        hotel_class: Check for suitable value of filters/hotel_class field (separated by comma to specify multiple values) returned in hotel-filters/list endpoint
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/hotels/list-in-boundary"
    querystring = {'tr_latitude': tr_latitude, 'tr_longitude': tr_longitude, 'bl_longitude': bl_longitude, 'bl_latitude': bl_latitude, }
    if pricesmax:
        querystring['pricesmax'] = pricesmax
    if amenities:
        querystring['amenities'] = amenities
    if pricesmin:
        querystring['pricesmin'] = pricesmin
    if offset:
        querystring['offset'] = offset
    if rooms:
        querystring['rooms'] = rooms
    if subcategory:
        querystring['subcategory'] = subcategory
    if currency:
        querystring['currency'] = currency
    if adults:
        querystring['adults'] = adults
    if child_rm_ages:
        querystring['child_rm_ages'] = child_rm_ages
    if zff:
        querystring['zff'] = zff
    if limit:
        querystring['limit'] = limit
    if hotel_class:
        querystring['hotel_class'] = hotel_class
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_search_deprecating(query: str, limit: int=30, sort: str='relevance', offset: int=0, lang: str='en_US', location_id: int=1, units: str='km', currency: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for related cities, countries, and suggestions"
    query: Name of cities, districts, places, etc...
        limit: The number of items per response (max 30)
        sort: One of the followings : relevance|distance
        offset: The number of items to ignore for paging purpose
        lang: The language code
        location_id: The value of location_id field returned right in this endpoint or .../locations/auto-complete endpoint
        units: One of the followings : km|mi
        currency: The currency code
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/locations/search"
    querystring = {'query': query, }
    if limit:
        querystring['limit'] = limit
    if sort:
        querystring['sort'] = sort
    if offset:
        querystring['offset'] = offset
    if lang:
        querystring['lang'] = lang
    if location_id:
        querystring['location_id'] = location_id
    if units:
        querystring['units'] = units
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_auto_complete_deprecating(query: str, lang: str='en_US', units: str='km', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List suggested locations by term or phrase"
    query: Name of cities, districts, places, etc...
        lang: The language code
        units: One of the followings : km|mi
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/locations/auto-complete"
    querystring = {'query': query, }
    if lang:
        querystring['lang'] = lang
    if units:
        querystring['units'] = units
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_v2_auto_complete(query: str, units: str='km', lang: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List suggested locations by term or phrase"
    query: Name of cities, districts, places, etc...
        units: One of the followings : km|mi
        lang: The language code
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/locations/v2/auto-complete"
    querystring = {'query': query, }
    if units:
        querystring['units'] = units
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flights_poll_deprecating(sid: str, currency: str='USD', mc: str=None, o: int=0, so: str='PRICE', ca: str=None, am: str=None, n: int=15, ns: str='NON_STOP,ONE_STOP', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "* TripAdvisor stops providing "Flights feature" via their mobile application, the web version is going to be down in the future.
		Poll for more available flight data until the value of "summary/c" field returned right in this endpoint is true. The flight APIs need to use as following : Firstly, you init a search session by using the create-session endpoint and get sid value Secondly, you repeatedly call poll endpoint until the summary/c field is true to get final total flight by summary/nr value. o parameter must be 0 otherwise the summary field is not returned. Thirdly, you repeatedly call poll endpoint with o increasing by n value step by step until you reach total flight."
    sid: The value of sid returned in flights/create-session endpoint
        currency: The currency code
        mc: Check for suitable value of "summary/cp" or "summary/ocp" field (separated by comma to specify multiple values) returned in flights/create-session or right in this endpoint
        o: The offset of items to be ignored in response for paging. Use summary/nr field returned in response to get total flights
        so: Check for suitable value of "so" field returned in flights/create-session or right in this endpoint
        ca: Check for suitable value of "summary/ap" field (separated by comma to specify multiple values) returned in flights/create-session or right in this endpoint
        am: Check for suitable value of "itineraries/ac" field (separated by comma to specify multiple values) returned in flights/create-session or right in this endpoint
        n: The number of items per response (max 15)
        ns: One of following NON&#95;STOP | ONE&#95;STOP | TWO&#95;PLUS or separated by comma to specify multiple values
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/flights/poll"
    querystring = {'sid': sid, }
    if currency:
        querystring['currency'] = currency
    if mc:
        querystring['mc'] = mc
    if o:
        querystring['o'] = o
    if so:
        querystring['so'] = so
    if ca:
        querystring['ca'] = ca
    if am:
        querystring['am'] = am
    if n:
        querystring['n'] = n
    if ns:
        querystring['ns'] = ns
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def flights_get_booking_url_deprecating(searchid: str, orig: str, is_id: str, searchhash: str, dest: str, impressionid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "* TripAdvisor stops providing "Flights feature" via their mobile application, the web version is going to be down in the future.
		Generate booking url relating to a specific flight"
    searchid: The value of search_params/sid field returned in response of .../flights/create-session endpoint. For example : 5aaa20e5-d5c8-4cdd-a657-b6453bb80756.1483
        orig: Airport code of origin. For example : DMK
        id: The value of itineraries/l/id field returned in response of .../flights/poll endpoint. For example : AgodaFlights|1|68
        searchhash: The value of summary/sh field returned in response of .../flights/create-session endpoint. For example : 2188ff0ed4e8e4ee7a50dc32b229e11d
        dest: Airport code of destination. For example : NYC
        impressionid: The value of itineraries/l/impressionId field returned in response of .../flights/poll endpoint. For example : 372e5654-b873-4c26-859b-633c29f7ad84.93175
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/flights/get-booking-url"
    querystring = {'searchId': searchid, 'Orig': orig, 'id': is_id, 'searchHash': searchhash, 'Dest': dest, }
    if impressionid:
        querystring['impressionId'] = impressionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_list_deprecated(location_id: int, lang: str='en_US', currency: str='USD', offset: int=None, limit: int=20, keyword: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List reviews related to a location by its location_id"
    location_id: The value of location_id field that returned in hotels/list, restaurants/list, or attractions/list endpoints
        lang: The language code
        currency: The currency code
        offset: The number of items to ignore for paging purpose
        limit: The number of items per response (max 20)
        keyword: Check for suitable value of text field returned in keywords/list endpoint
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/reviews/list"
    querystring = {'location_id': location_id, }
    if lang:
        querystring['lang'] = lang
    if currency:
        querystring['currency'] = currency
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if keyword:
        querystring['keyword'] = keyword
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def attractions_list_by_latlng_deprecated(latitude: int, longitude: int, currency: str='USD', lang: str='en_US', distance: int=None, limit: int=None, offset: int=None, lunit: str='km', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List attractions by specifying an coordinate and radius"
    latitude: Latitude of coordinate
        longitude: Longitude of coordinate
        currency: The currency code
        lang: The language code
        distance: The radius around specified coordinate (max 25)
        limit: The number of items per response (max 30)
        offset: The number of items to ignore for paging purpose
        lunit: One of the followings km|mi
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/attractions/list-by-latlng"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    if currency:
        querystring['currency'] = currency
    if lang:
        querystring['lang'] = lang
    if distance:
        querystring['distance'] = distance
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if lunit:
        querystring['lunit'] = lunit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def attractions_list_in_boundary_deprecated(bl_latitude: int, tr_latitude: int, bl_longitude: int, tr_longitude: int, currency: str='USD', offset: int=None, lang: str='en_US', limit: int=None, lunit: str='km', min_rating: int=None, bookable_first: bool=None, subcategory: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List attractions by specifying coordinates of bottom left and top right of a boundary"
    bl_latitude: Latitude of bottom left coordinate
        tr_latitude: Latitude of top right coordinate
        bl_longitude: Longitude of bottom left coordinate
        tr_longitude: Longitude of top right coordinate
        currency: The currency code
        offset: The number of items to ignore for paging purpose
        lang: The language code
        limit: The number of items per response (max 30)
        lunit: One of the followings km|mi
        min_rating: Rating - Min 3 max 5
        bookable_first: Book online first
        subcategory: Attraction category - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (only one value is allowed at a time) returned right in this endpoint
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/attractions/list-in-boundary"
    querystring = {'bl_latitude': bl_latitude, 'tr_latitude': tr_latitude, 'bl_longitude': bl_longitude, 'tr_longitude': tr_longitude, }
    if currency:
        querystring['currency'] = currency
    if offset:
        querystring['offset'] = offset
    if lang:
        querystring['lang'] = lang
    if limit:
        querystring['limit'] = limit
    if lunit:
        querystring['lunit'] = lunit
    if min_rating:
        querystring['min_rating'] = min_rating
    if bookable_first:
        querystring['bookable_first'] = bookable_first
    if subcategory:
        querystring['subcategory'] = subcategory
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def attractions_get_details_deprecated(location_id: int, currency: str='USD', lang: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all information of specific attracting location by its location_id"
    location_id: The value of location_id field that returned in attractions/list endpoint
        currency: The currency code
        lang: The language code
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/attractions/get-details"
    querystring = {'location_id': location_id, }
    if currency:
        querystring['currency'] = currency
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def restaurants_list_deprecated(location_id: int, lunit: str='km', offset: int=None, min_rating: str=None, limit: int=30, restaurant_mealtype: str=None, lang: str='en_US', restaurant_tagcategory_standalone: str='10591', restaurant_styles: str=None, currency: str='USD', combined_food: str=None, prices_restaurants: str=None, dietary_restrictions: str=None, restaurant_tagcategory: str='10591', open_now: bool=None, restaurant_dining_options: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List restaurants related to a location by location_id"
    location_id: The value of location_id field that returned in locations/search endpoint
        lunit: One of the followings km|mi
        offset: The number of items to ignore for paging purpose
        min_rating: Min 3 - Max 5
        limit: The number of items per response (max 30)
        restaurant_mealtype: Meals - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to filter by multiple values. Ex : 10598,10599) returned right in this endpoint
        lang: The language code
        restaurant_tagcategory_standalone: Establishment Type - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        restaurant_styles: Restaurant Features - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        currency: The currency code
        combined_food: Cuisine Type - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        prices_restaurants: Prices - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        dietary_restrictions: Dietary Restrictions - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        restaurant_tagcategory: Establishment Type - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        open_now: Only returns restaurants which are opening now
        restaurant_dining_options: Restaurant Features - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/restaurants/list"
    querystring = {'location_id': location_id, }
    if lunit:
        querystring['lunit'] = lunit
    if offset:
        querystring['offset'] = offset
    if min_rating:
        querystring['min_rating'] = min_rating
    if limit:
        querystring['limit'] = limit
    if restaurant_mealtype:
        querystring['restaurant_mealtype'] = restaurant_mealtype
    if lang:
        querystring['lang'] = lang
    if restaurant_tagcategory_standalone:
        querystring['restaurant_tagcategory_standalone'] = restaurant_tagcategory_standalone
    if restaurant_styles:
        querystring['restaurant_styles'] = restaurant_styles
    if currency:
        querystring['currency'] = currency
    if combined_food:
        querystring['combined_food'] = combined_food
    if prices_restaurants:
        querystring['prices_restaurants'] = prices_restaurants
    if dietary_restrictions:
        querystring['dietary_restrictions'] = dietary_restrictions
    if restaurant_tagcategory:
        querystring['restaurant_tagcategory'] = restaurant_tagcategory
    if open_now:
        querystring['open_now'] = open_now
    if restaurant_dining_options:
        querystring['restaurant_dining_options'] = restaurant_dining_options
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def restaurants_list_by_latlng_deprecated(latitude: int, longitude: int, offset: int=None, distance: int=2, restaurant_styles: str=None, min_rating: str=None, restaurant_tagcategory_standalone: str=None, restaurant_mealtype: str=None, prices_restaurants: str=None, currency: str='USD', open_now: bool=None, limit: int=30, lang: str='en_US', combined_food: str=None, lunit: str='km', dietary_restrictions: str=None, restaurant_dining_options: str=None, restaurant_tagcategory: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List restaurants by specifying an coordinate and radius"
    latitude: Latitude of coordinate
        longitude: Longitude of coordinate
        offset: The number of items to ignore for paging purpose
        distance: The radius around specified coordinate (max 10)
        restaurant_styles: Restaurant Features - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        min_rating: Min 3 - Max 5
        restaurant_tagcategory_standalone: Establishment Type - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        restaurant_mealtype: Meals - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        prices_restaurants: Prices - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        currency: The currency code
        open_now: Only returns restaurants which are opening now
        limit: The number of items per response (max 30)
        lang: The language code
        combined_food: Cuisine Type - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        lunit: One of the followings km|mi
        dietary_restrictions: Dietary Restrictions - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        restaurant_dining_options: Restaurant Features - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        restaurant_tagcategory: Establishment Type - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/restaurants/list-by-latlng"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    if offset:
        querystring['offset'] = offset
    if distance:
        querystring['distance'] = distance
    if restaurant_styles:
        querystring['restaurant_styles'] = restaurant_styles
    if min_rating:
        querystring['min_rating'] = min_rating
    if restaurant_tagcategory_standalone:
        querystring['restaurant_tagcategory_standalone'] = restaurant_tagcategory_standalone
    if restaurant_mealtype:
        querystring['restaurant_mealtype'] = restaurant_mealtype
    if prices_restaurants:
        querystring['prices_restaurants'] = prices_restaurants
    if currency:
        querystring['currency'] = currency
    if open_now:
        querystring['open_now'] = open_now
    if limit:
        querystring['limit'] = limit
    if lang:
        querystring['lang'] = lang
    if combined_food:
        querystring['combined_food'] = combined_food
    if lunit:
        querystring['lunit'] = lunit
    if dietary_restrictions:
        querystring['dietary_restrictions'] = dietary_restrictions
    if restaurant_dining_options:
        querystring['restaurant_dining_options'] = restaurant_dining_options
    if restaurant_tagcategory:
        querystring['restaurant_tagcategory'] = restaurant_tagcategory
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def restaurants_get_details_deprecated(location_id: int, currency: str='USD', lang: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all information of a specific restaurant by its location_id"
    location_id: The value of location_id field that returned in restaurants/list endpoint
        currency: The currency code
        lang: The language code
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/restaurants/get-details"
    querystring = {'location_id': location_id, }
    if currency:
        querystring['currency'] = currency
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def restaurants_list_in_boundary_deprecated(tr_longitude: int, tr_latitude: int, bl_latitude: int, bl_longitude: int, prices_restaurants: str=None, min_rating: int=None, dietary_restrictions: str=None, open_now: bool=None, offset: int=None, restaurant_dining_options: str=None, limit: int=30, combined_food: str=None, restaurant_tagcategory: str='10591', restaurant_mealtype: str=None, currency: str='USD', restaurant_styles: str=None, lunit: str='km', restaurant_tagcategory_standalone: str='10591', lang: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List restaurants by specifying coordinates of bottom left and top right of a boundary"
    tr_longitude: Longitude of top right coordinate
        tr_latitude: Latitude of top right coordinate
        bl_latitude: Latitude of bottom left coordinate
        bl_longitude: Longitude of bottom left coordinate
        prices_restaurants: Prices - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        min_rating: Min 3 - Max 5
        dietary_restrictions: Dietary Restrictions - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        open_now: Only returns restaurants which are opening now
        offset: The number of items to ignore for paging purpose
        restaurant_dining_options: Restaurant Features - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        limit: The number of items per response (max 30)
        combined_food: Cuisine Type - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        restaurant_tagcategory: Establishment Type - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        restaurant_mealtype: Meals - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to filter by multiple values. Ex : 10598,10599) returned right in this endpoint
        currency: The currency code
        restaurant_styles: Restaurant Features - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        lunit: One of the followings km|mi
        restaurant_tagcategory_standalone: Establishment Type - Check for suitable values of filters&#95;v2/filter&#95;sections/filter&#95;groups/options/value field (separated by comma to specify multiple values) returned right in this endpoint
        lang: The language code
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/restaurants/list-in-boundary"
    querystring = {'tr_longitude': tr_longitude, 'tr_latitude': tr_latitude, 'bl_latitude': bl_latitude, 'bl_longitude': bl_longitude, }
    if prices_restaurants:
        querystring['prices_restaurants'] = prices_restaurants
    if min_rating:
        querystring['min_rating'] = min_rating
    if dietary_restrictions:
        querystring['dietary_restrictions'] = dietary_restrictions
    if open_now:
        querystring['open_now'] = open_now
    if offset:
        querystring['offset'] = offset
    if restaurant_dining_options:
        querystring['restaurant_dining_options'] = restaurant_dining_options
    if limit:
        querystring['limit'] = limit
    if combined_food:
        querystring['combined_food'] = combined_food
    if restaurant_tagcategory:
        querystring['restaurant_tagcategory'] = restaurant_tagcategory
    if restaurant_mealtype:
        querystring['restaurant_mealtype'] = restaurant_mealtype
    if currency:
        querystring['currency'] = currency
    if restaurant_styles:
        querystring['restaurant_styles'] = restaurant_styles
    if lunit:
        querystring['lunit'] = lunit
    if restaurant_tagcategory_standalone:
        querystring['restaurant_tagcategory_standalone'] = restaurant_tagcategory_standalone
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotels_list_deprecated(adults: int, nights: int, location_id: int, rooms: int, sort: str='recommended', offset: int=0, order: str='asc', pricesmax: int=None, limit: int=30, child_rm_ages: str=None, lang: str='en_US', subcategory: str=None, zff: str=None, checkin: str=None, currency: str='USD', pricesmin: int=None, hotel_class: str=None, amenities: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all hotels around the world with options and filters"
    adults: The number of adults in all rooms
        nights: The number of nights to live
        location_id: The value of location_id field that returned in locations/search endpoint
        rooms: The number of rooms
        sort: One of the followings : recommended|popularity|price
        offset: The number of items to ignore for paging purpose
        order: One of the followings : asc|desc, this param is used with sort by price
        pricesmax: Check for suitable price range in filters/prices_slider field returned in hotel-filters/list endpoint. For exmaple : $120 -> 120
        limit: The number of items per response (max 30)
        child_rm_ages: The age of every children separated by comma in all rooms. Ex : 7,10
        lang: The language code
        subcategory: Check for suitable value of filters/subcategory field (separated by comma to filter by multiple values. Ex : hotel,bb,specialty) returned in hotel-filters/list endpoint
        zff: Hotel Style - Check for suitable value of filters/zff field (separated by comma to filter by multiple values. Ex : 4,6) returned in hotel-filters/list endpoint
        checkin: The check-in date at hotel, the format is yyyy-MM-dd. Ex : 2020-05-15
        currency: The currency code
        pricesmin: Check for suitable price range in filters/prices_slider field returned in hotel-filters/list endpoint. For exmaple : $120 -> 120
        hotel_class: Check for suitable value of filters/hotel_class field (separated by comma to filter by multiple values. Ex : 1,2,3) returned in hotel-filters/list endpoint
        amenities: Check for suitable value of filters/amenities field (separated by comma to filter by multiple values. Ex : beach,bar&#95;lounge,airport&#95;transportation) returned in hotel-filters/list endpoint
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/hotels/list"
    querystring = {'adults': adults, 'nights': nights, 'location_id': location_id, 'rooms': rooms, }
    if sort:
        querystring['sort'] = sort
    if offset:
        querystring['offset'] = offset
    if order:
        querystring['order'] = order
    if pricesmax:
        querystring['pricesmax'] = pricesmax
    if limit:
        querystring['limit'] = limit
    if child_rm_ages:
        querystring['child_rm_ages'] = child_rm_ages
    if lang:
        querystring['lang'] = lang
    if subcategory:
        querystring['subcategory'] = subcategory
    if zff:
        querystring['zff'] = zff
    if checkin:
        querystring['checkin'] = checkin
    if currency:
        querystring['currency'] = currency
    if pricesmin:
        querystring['pricesmin'] = pricesmin
    if hotel_class:
        querystring['hotel_class'] = hotel_class
    if amenities:
        querystring['amenities'] = amenities
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotels_list_by_latlng_deprecated(latitude: int, longitude: int, adults: int=1, offset: int=None, child_rm_ages: str='7,10', currency: str='USD', rooms: int=1, distance: int=None, lang: str='en_US', pricesmax: int=None, hotel_class: str='1,2,3', limit: int=30, zff: str='4,6', amenities: str='beach,bar_lounge,airport_transportation', nights: str='2', subcategory: str='hotel,bb,specialty', checkin: str='2022-03-08', pricesmin: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List hotels by specifying a coordinate and radius"
    longitude: Longitude of coordinate
        adults: The number of adults in all rooms
        offset: The number of items to ignore for paging purpose
        child_rm_ages: The age of every children separated by comma in all rooms
        currency: The currency code
        rooms: The number of rooms
        distance: The radius to look for nearest places
        lang: The language code
        pricesmax: Check for suitable price range in filters/prices_slider field returned in hotel-filters/list endpoint. For exmaple : $120 -> 120
        hotel_class: Check for suitable value of filters/hotel_class field (separated by comma to specify multiple values) returned in hotel-filters/list endpoint
        limit: The number of items per response (max 30)
        zff: Hotel Style - Check for suitable value of \\\"filters/zff\\\" field (separated by comma to specify multiple values) returned in hotel-filters/list endpoint
        amenities: Check for suitable value of filters/amenities field (separated by comma to specify multiple values) returned in hotel-filters/list endpoint
        nights: The number of nights to live
        subcategory: Check for suitable value of filters/subcategory field (separated by comma to specify multiple values) returned in hotel-filters/list endpoint
        checkin: The check-in date at hotel
        pricesmin: Check for suitable price range in filters/prices_slider field returned in hotel-filters/list endpoint. For exmaple : $10 -> 10
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/hotels/list-by-latlng"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    if adults:
        querystring['adults'] = adults
    if offset:
        querystring['offset'] = offset
    if child_rm_ages:
        querystring['child_rm_ages'] = child_rm_ages
    if currency:
        querystring['currency'] = currency
    if rooms:
        querystring['rooms'] = rooms
    if distance:
        querystring['distance'] = distance
    if lang:
        querystring['lang'] = lang
    if pricesmax:
        querystring['pricesmax'] = pricesmax
    if hotel_class:
        querystring['hotel_class'] = hotel_class
    if limit:
        querystring['limit'] = limit
    if zff:
        querystring['zff'] = zff
    if amenities:
        querystring['amenities'] = amenities
    if nights:
        querystring['nights'] = nights
    if subcategory:
        querystring['subcategory'] = subcategory
    if checkin:
        querystring['checkin'] = checkin
    if pricesmin:
        querystring['pricesmin'] = pricesmin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotels_get_details_deprecated(location_id: int, nights: int=2, child_rm_ages: str=None, lang: str='en_US', checkin: str='2022-03-15', rooms: str=None, adults: int=1, currency: str='USD', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all information of hotels"
    location_id: The value of location_id field that returned in .../hotels/list... endpoint
        nights: The number of nights to live
        child_rm_ages: The age of every children separated by comma in all rooms
        lang: The language code
        checkin: The check-in date at hotel
        rooms: The number of rooms
        adults: The number of adults in all rooms
        currency: The currency code
        
    """
    url = f"https://travel-advisor.p.rapidapi.com/hotels/get-details"
    querystring = {'location_id': location_id, }
    if nights:
        querystring['nights'] = nights
    if child_rm_ages:
        querystring['child_rm_ages'] = child_rm_ages
    if lang:
        querystring['lang'] = lang
    if checkin:
        querystring['checkin'] = checkin
    if rooms:
        querystring['rooms'] = rooms
    if adults:
        querystring['adults'] = adults
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

