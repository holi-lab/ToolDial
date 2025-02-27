import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def listing_reviews(is_id: str, date_time: str='2023-01-11 10:52:50', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Return up to 20 reviews after the specified datetime. Each reviews contains those fields: review_id, comments, response, date_time, language, rating: 5, guest_id. Available only for properties. Available only for the properties whose prices or availability have been requested via API at least one time in the past and at least 5 hours before actual time."
    id: the listing id
        date_time: the month
        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/listingReviews"
    querystring = {'id': is_id, }
    if date_time:
        querystring['date_time'] = date_time
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def room_prices_and_availability_by_lat_lng(year: int, lat: int, lng: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve Rooms average Price, average price of available rooms only, availability percent and number of processed rooms in a range from a given geographical point. Max range is 20000 meters."
    year: the year
        lat: latitude
        lng: longitude
        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/avgRoomPricesByLatLng"
    querystring = {'year': year, 'lat': lat, 'lng': lng, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rooms_prices_and_availability_by_administrative_divisions(country_code: str, year: int, admin1: str='Lazio', admin2: str='Citta metropolitana di Roma Capitale', admin4: str='Municipio Roma I', place: str='Trastevere', admin3: str='Roma', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve average rooms Price, average price of available rooms only, availability percent and number of processed rooms in an geographical administrative division. Geographic area names can be found from "Get administrative divisions" endpoint. admin* parameters are optional but to use admin2, admin1 is required, to use admin 3, admin2 and admin1 are required and so on. Not respecting this requirement could give you unwanted results (ie rooms with cities with the same name but in different countries)."
    
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/avgRoomPricesByAdmins"
    querystring = {'country_code': country_code, 'year': year, }
    if admin1:
        querystring['admin1'] = admin1
    if admin2:
        querystring['admin2'] = admin2
    if admin4:
        querystring['admin4'] = admin4
    if place:
        querystring['place'] = place
    if admin3:
        querystring['admin3'] = admin3
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listings_by_market(market: str, offset: int, maxguestcapacity: int=2, bedrooms: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve listings ids and last updated dates for prices, availability and ratings in the specified market. Returns 50 results. Can be filtered on bedrooms and max guest capacity"
    offset: index to start from
        maxguestcapacity: Max guest the listing can accomodate
        bedrooms: number of bedrooms
        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/listingsByMarket"
    querystring = {'market': market, 'offset': offset, }
    if maxguestcapacity:
        querystring['maxGuestCapacity'] = maxguestcapacity
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listings_by_lat_lng(lat: int, lng: int, offset: int, range: int, bedrooms: int=1, maxguestcapacity: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve listings ids, distance from starting point and last updated dates for prices, availability and ratings in a range from a given geographical point. Returns 50 results. Can be filtered on bedrooms and max guest capacity"
    lat: latitude
        lng: longitude
        offset: index to start from
        range: range in meters from latitude and longitude point
        bedrooms: number of bedrooms
        maxguestcapacity: Max guest the listing can accomodate
        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/listingsByLatLng"
    querystring = {'lat': lat, 'lng': lng, 'offset': offset, 'range': range, }
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if maxguestcapacity:
        querystring['maxGuestCapacity'] = maxguestcapacity
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def count_listings_by_market(market: str, bedrooms: int=1, maxguestcapacity: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve listings count in the specified market. Can be filtered on bedrooms and max guest capacity"
    bedrooms: Number of bedrooms
        maxguestcapacity: Max guest the listing can accomodate
        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/countByMarket"
    querystring = {'market': market, }
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if maxguestcapacity:
        querystring['maxGuestCapacity'] = maxguestcapacity
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listings_by_administrative_divisions(offset: str, state: str, admin3: str='Roma', admin1: str='Lazio', admin4: str='Municipio Roma I', bedrooms: str='2', maxguestcapacity: str='4', place: str='Borgo', admin2: str='Citta metropolitana di Roma Capitale', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve listings ids and last updated dates for prices, availability and ratings in the specified geographic area. Geocrafic area names can be found from "Get administrative divisions" endpoint. admin* parameters are optional but to use admin2, admin1 is required, to use admin 3, admin2 and admin1 are required and so on. Not respecting this requirement could give you unwanted results (ie listings with cities with the same name but in different countries). Returns 50 results. Can be filtered on bedrooms and max guest capacity"
    
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/listingsByGeoRef"
    querystring = {'offset': offset, 'state': state, }
    if admin3:
        querystring['admin3'] = admin3
    if admin1:
        querystring['admin1'] = admin1
    if admin4:
        querystring['admin4'] = admin4
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if maxguestcapacity:
        querystring['maxGuestCapacity'] = maxguestcapacity
    if place:
        querystring['place'] = place
    if admin2:
        querystring['admin2'] = admin2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def count_listings_by_lat_lng(lng: int, range: int, lat: int, maxguestcapacity: int=2, bedrooms: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve listings count in a range from a given geographical point. Can be filtered on bedrooms and max guest capacity"
    lng: longitude
        range: range in meters from latitude and longitude point
        lat: latitude
        maxguestcapacity: Max guest the listing can accomodate
        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/countByLatLng"
    querystring = {'lng': lng, 'range': range, 'lat': lat, }
    if maxguestcapacity:
        querystring['maxGuestCapacity'] = maxguestcapacity
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def count_listings_by_administrative_divisions(state: str, admin1: str='Lazio', admin2: str='Citta metropolitana di Roma Capitale', admin3: str='Roma', place: str='Borgo', admin4: str='Municipio Roma I', maxguestcapacity: str='4', bedrooms: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve listings count in the specified geographic area. Geocrafic area names can be found from "Get administrative divisions" endpoint. admin* parameters are optional but to use admin2, admin1 is required, to use admin 3, admin2 and admin1 are required and so on. Not respecting this requirement could give you unwanted results (ie listings with cities with the same name but in different countries). Can be filtered on bedrooms and max guest capacity"
    
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/countByGeoRef"
    querystring = {'state': state, }
    if admin1:
        querystring['admin1'] = admin1
    if admin2:
        querystring['admin2'] = admin2
    if admin3:
        querystring['admin3'] = admin3
    if place:
        querystring['place'] = place
    if admin4:
        querystring['admin4'] = admin4
    if maxguestcapacity:
        querystring['maxGuestCapacity'] = maxguestcapacity
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listing_details(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve listing details."
    id: the listing Id
        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/listing"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listing_prices_full(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve listing prices for the next 12 months"
    id: the listing id

        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/listingPricesFull"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listing_availability_full(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve listing availability and stay length rules for the next 12 months
		**Note**: The fact the available field is true does not means that the listing can be booked. It only tells indicates that it is not taken. To confirm if it is really available for stay you must verify all previous and following day rules for stay length are respected."
    id: the listing id

        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/listingAvailabilityFull"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listing_availability(year: int, is_id: str, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve listing availability and stay length rules for the requested month.
		**Note**: The fact the available field is true does not means that the listing can be booked. It only tells indicates that it is not taken. To confirm if it is really available for stay you must verify all previous and following day rules for stay length are respected."
    year: the year
        id: the listing id

        month: the month
        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/listingavailability"
    querystring = {'year': year, 'id': is_id, 'month': month, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listing_status_full(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the calculated status (available/not available for stay) for the next 12 months of a listing based on previous and following days stay rules."
    id: the listing id
        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/listingStatusFull"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listing_prices(is_id: str, month: str, year: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve listing prices for the requested month"
    id: the listing id

        month: the month
        year: the year
        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/listingPrices"
    querystring = {'id': is_id, 'month': month, 'year': year, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listing_status(is_id: str, year: int, month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve the calculated status (available/not available for stay) of a listing based on previous and following days stay rules."
    id: the listing id
        year: the year
        month: the month
        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/listingstatus"
    querystring = {'id': is_id, 'year': year, 'month': month, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prices_and_availability_by_administrative_divisions(month: str, country_code: str, year: int, admin3: str='Roma', bedrooms: str='2', admin1: str='Lazio', admin2: str='Citta metropolitana di Roma Capitale', maxguestcapacity: int=4, place: str='Trastevere', admin4: str='Municipio Roma I', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve average Price, average price of available properties only, availability percent and number of processed properties in an geographical administrative division. Geographic area names can be found from "Get administrative divisions" endpoint. admin* parameters are optional but to use admin2, admin1 is required, to use admin 3, admin2 and admin1 are required and so on. Not respecting this requirement could give you unwanted results (ie listings with cities with the same name but in different countries). Can be filtered on bedrooms and max guest capacity."
    
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/avgPricesByAdmins"
    querystring = {'month': month, 'country_code': country_code, 'year': year, }
    if admin3:
        querystring['admin3'] = admin3
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if admin1:
        querystring['admin1'] = admin1
    if admin2:
        querystring['admin2'] = admin2
    if maxguestcapacity:
        querystring['maxGuestCapacity'] = maxguestcapacity
    if place:
        querystring['place'] = place
    if admin4:
        querystring['admin4'] = admin4
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_administrative_divisions(countrycode: str, admin2: str='RM', admin1: str='07', admin4: str='05809101', admin3: str='058091', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve geographical admin names to be used in Listing by georef endpoint. admin* parameters are optional but to use admin2, admin1 is required, to use admin 3, admin2 and admin1 are required and so on. Not respecting this requirement could give you unwanted results (ie cities with the same name but in different countries). Returns 50 results"
    
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/getadmins"
    querystring = {'countrycode': countrycode, }
    if admin2:
        querystring['admin2'] = admin2
    if admin1:
        querystring['admin1'] = admin1
    if admin4:
        querystring['admin4'] = admin4
    if admin3:
        querystring['admin3'] = admin3
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def prices_and_availability_by_lat_lng(year: int, lat: int, lng: int, range: int, month: str, bedrooms: int=1, maxguestcapacity: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve average Price, average price of available properties only, availability percent and number of processed properties in a range from a given geographical point. Max range is 20000 meters. Can be filtered on bedrooms and max guest capacity."
    year: the year
        lat: latitude
        lng: longitude
        range: range in meters from latitude and longitude point
        month: the month
        bedrooms: number of bedrooms
        maxguestcapacity: Max guest the listing can accomodate
        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/avgPricesByLatLng"
    querystring = {'year': year, 'lat': lat, 'lng': lng, 'range': range, 'month': month, }
    if bedrooms:
        querystring['bedrooms'] = bedrooms
    if maxguestcapacity:
        querystring['maxGuestCapacity'] = maxguestcapacity
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def amenities(offset: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves amenities ids And descriptions. Returns 50 results"
    offset: index to start from
        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/amenities"
    querystring = {'offset': offset, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def markets(offset: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve Market names to be used in Listing by market endpoint. Returns 50 results"
    offset: index to start from
        
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2/markets"
    querystring = {'offset': offset, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def connect_test(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "No parameters needed. Useful to test connectivity and authentication"
    
    """
    url = f"https://airbnb-listings.p.rapidapi.com/v2"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "airbnb-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

