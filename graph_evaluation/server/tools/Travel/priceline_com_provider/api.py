import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def auto_suggest(string: str, sort: str=None, get_airports: bool=None, order: str=None, combine_regions: bool=None, get_pois: bool=None, get_regions: bool=None, get_cities: bool=None, show_all_cities: bool=None, max_results: int=None, get_hotels: bool=None, spellcheck: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API will provide a list of possible cities and hotels for a given search string"
    string: Search string that will enable a list of selection to be listed to the traveller.
        sort: Enum: rank, name. Method of sorting the results. Valid options: rank, name
        get_airports: Include airports in search results. Valid Options: True or False.
        order: Method of ordering the results of the search. Valid options: asc or desc.
        combine_regions: Enables the spell check option for the search string using either true or false.
        get_pois: Include Points of Interest in search results. Valid Options: True or False
        get_regions: Include Regions in search results. Valid Options: True or False.
        get_cities: Include cities in search results. Valid Options: True or False.
        show_all_cities: Will filter out cities with no hotels. Valid Options: False = filter out cities without hotels, True = show cities with and without hotels.
        max_results: Number passed is the maximum number of results returned.
        get_hotels: Include hotels in search results. Valid Options: True or False.
        spellcheck: Enables the spell check option for the search string using either true or false.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/autoSuggest"
    querystring = {'string': string, }
    if sort:
        querystring['sort'] = sort
    if get_airports:
        querystring['get_airports'] = get_airports
    if order:
        querystring['order'] = order
    if combine_regions:
        querystring['combine_regions'] = combine_regions
    if get_pois:
        querystring['get_pois'] = get_pois
    if get_regions:
        querystring['get_regions'] = get_regions
    if get_cities:
        querystring['get_cities'] = get_cities
    if show_all_cities:
        querystring['show_all_cities'] = show_all_cities
    if max_results:
        querystring['max_results'] = max_results
    if get_hotels:
        querystring['get_hotels'] = get_hotels
    if spellcheck:
        querystring['spellcheck'] = spellcheck
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_results_v(dropoff_time: str, pickup_time: str, dropoff_date: str, pickup_date: str, dropoff_lat_long: str=None, currency: str=None, pickup_code: str='JFK', pickup_counters: str=None, prepaid_rates: bool=None, dropoff_code: str='JFK', dropoff_counters: str=None, pickup_lat_long: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The getResultsRequest endpoint returns car availability based on the provided search criteria"
    dropoff_time: Car Drop Off Time (HH:mm:ss)
        pickup_time: Car Pick Up Time (HH:mm:ss)
        dropoff_date: Car Drop Off Date (YYYY-MM-DD or MM/DD/YYYY).
        pickup_date: Car Pick Up Date (YYYY-MM-DD or MM/DD/YYYY)
        dropoff_lat_long: dropoff location by latitude and longitude. Accepts a comma separated latitude and longitude values.
        currency: Requested currency for the results. ISO 4217 format.
        pickup_code: Accepts a 3-character IATA airport code or a single PPN City ID.
        pickup_counters: Accepts an array of one or more keys as a 2-character car company code and the values as the counter location code.
        prepaid_rates: If activated, set the value as prepaid_rates=1 to display prepaid rates
        dropoff_code: Accepts a 3-character IATA airport code or a single PPN City ID.
        dropoff_counters: Accepts an array of one or more keys as a 2-character car company code and the values as the counter ocation code.
        pickup_lat_long: Pickup location by latitude and longitude. Accepts a comma separated latitude and longitude values.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/cars/resultsVer"
    querystring = {'dropoff_time': dropoff_time, 'pickup_time': pickup_time, 'dropoff_date': dropoff_date, 'pickup_date': pickup_date, }
    if dropoff_lat_long:
        querystring['dropoff_lat_long'] = dropoff_lat_long
    if currency:
        querystring['currency'] = currency
    if pickup_code:
        querystring['pickup_code'] = pickup_code
    if pickup_counters:
        querystring['pickup_counters'] = pickup_counters
    if prepaid_rates:
        querystring['prepaid_rates'] = prepaid_rates
    if dropoff_code:
        querystring['dropoff_code'] = dropoff_code
    if dropoff_counters:
        querystring['dropoff_counters'] = dropoff_counters
    if pickup_lat_long:
        querystring['pickup_lat_long'] = pickup_lat_long
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_hotels_locations_by_geolocation(latitude: int, longitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search locations by coordinates. Set coordinates latitude and longitude"
    latitude: Latitude
        longitude: Longitude
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v1/hotels/locations-by-geo"
    querystring = {'latitude': latitude, 'longitude': longitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_cities(limit: int=100, resume_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads a list of Hotel cities"
    limit: Limits the number of results from the response.
        resume_key: Resume results from given ID.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/downloadCities"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if resume_key:
        querystring['resume_key'] = resume_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_companies(resume_key: str=None, limit: int=500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads a list of companies"
    resume_key: Resume results from given ID.
        limit: Limits the number of results from the response.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/cars/downloadCompanies"
    querystring = {}
    if resume_key:
        querystring['resume_key'] = resume_key
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_hotels_locations(name: str, search_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search locations by name"
    name: Name
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v1/hotels/locations"
    querystring = {'name': name, 'search_type': search_type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_express_results(check_out: str, check_in: str, airport_code: str=None, adults: int=None, sort_by: str=None, multiple_deals: bool=None, rate_limit: int=None, limit_to_country: bool=None, longitude: str=None, latitude: str=None, currency: str=None, limit: int=100, hotel_ids: str=None, city_id: str='800049480', language: str='fr-FR', children: int=None, country_code: str='BR', output_version: int=3, radius: int=None, sid: str='iSiX639', rooms: int=None, rate_identifier: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides discounted Express (Cached) and Closed User Group (Live) Rates using the getExpress.Results endpoint."
    check_out: Check In Date (YYYY-MM-DD or MM/DD/YYYY)
        check_in: Check In Date (YYYY-MM-DD or MM/DD/YYYY)
        airport_code: Accepts a 3-character IATA airport code.
        adults: The total number of adult occupants for all rooms requested. Used with children parameter to determine occupancy. Example: Two rooms, each with one adult and one child occupants, adults=2 and children=2
        sort_by: Sort results by a given option. Default sort is by guest_score. Valid Options: gs = guest_score, sr = star_rating, lp = lowest_price, hp = highest_price, ds = distance, mp = most_popular.
        multiple_deals: Multi Rates are provided Valid Options: 0 = false, 1 = true.
        rate_limit: Number passed to limit the number of rates returned. Defaults to returning all available rates
        limit_to_country: Limits results to country provided. Valid Options: true or false.
        longitude: Search for property availability around a specific longitude coordinate
        latitude: Search for property availability around a specific latitude coordinate.
        currency: Requested currency for the results. ISO 4217 format.
        limit: Limits the number of results from the response.
        hotel_ids: Comma separated string of PPN hotel ids (Semi Opaque Only)
        city_id: Accepts a single PPN City ID.)
        language: Language code: en-US, es-ES, fr-FR, pt-BR
        children: The total number of child occupants for all rooms requested. Used with adults parameter to determine occupancy. Example: Two rooms, each with one adult and one child occupants, adults=2 and children=2
        country_code: Pass the user s country to see rates with regional pricing. This is a two character ISO Alpha-2 country code.
        output_version: Enum: 1 2 3 4
        radius: Radius in miles the results are from
        sid: Session ID. Random string
        rooms: Number of rooms required for all occupants
        rate_identifier: A toggle to show if rate identifier is being passed. Valid Options: 0 = false, 1 = true. Rate is a string that is set for each hotel and holds all the information regarding the rate that we send to priceline.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/expressResults"
    querystring = {'check_out': check_out, 'check_in': check_in, }
    if airport_code:
        querystring['airport_code'] = airport_code
    if adults:
        querystring['adults'] = adults
    if sort_by:
        querystring['sort_by'] = sort_by
    if multiple_deals:
        querystring['multiple_deals'] = multiple_deals
    if rate_limit:
        querystring['rate_limit'] = rate_limit
    if limit_to_country:
        querystring['limit_to_country'] = limit_to_country
    if longitude:
        querystring['longitude'] = longitude
    if latitude:
        querystring['latitude'] = latitude
    if currency:
        querystring['currency'] = currency
    if limit:
        querystring['limit'] = limit
    if hotel_ids:
        querystring['hotel_ids'] = hotel_ids
    if city_id:
        querystring['city_id'] = city_id
    if language:
        querystring['language'] = language
    if children:
        querystring['children'] = children
    if country_code:
        querystring['country_code'] = country_code
    if output_version:
        querystring['output_version'] = output_version
    if radius:
        querystring['radius'] = radius
    if sid:
        querystring['sid'] = sid
    if rooms:
        querystring['rooms'] = rooms
    if rate_identifier:
        querystring['rate_identifier'] = rate_identifier
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_airports(resume_key: str=None, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads a list of Airports"
    resume_key: Resume results from given ID.
        limit: Limits the number of results from the response.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/downloadAirports"
    querystring = {}
    if resume_key:
        querystring['resume_key'] = resume_key
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_countries(resume_key: str=None, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads a list of countries"
    resume_key: Resume results from given ID.
        limit: Limits the number of results from the response.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/downloadCountries"
    querystring = {}
    if resume_key:
        querystring['resume_key'] = resume_key
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_details(hotel_id: int, offset_of_reviews: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all reviews and images of the hotel by hotel_id"
    hotel_id: Hotel id
        offset_of_reviews: Offset of reviews
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v1/hotels/details"
    querystring = {'hotel_id': hotel_id, }
    if offset_of_reviews:
        querystring['offset_of_reviews'] = offset_of_reviews
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_cars_locations(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search locations by name"
    name: Name
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v1/cars-rentals/locations"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_flights_locations(name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search airports and locations by name"
    name: Name
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v1/flights/locations"
    querystring = {'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def booking_details_of_the_hotel(date_checkin: str, hotel_id: int, date_checkout: str, rooms_number: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hotel descriptions, prices and available booking options. Indicate the hotel_id, check-in and check-out date"
    date_checkin: Checkin date
        hotel_id: Hotel id
        date_checkout: Checkout date
        rooms_number: Rooms number
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v1/hotels/booking-details"
    querystring = {'date_checkin': date_checkin, 'hotel_id': hotel_id, 'date_checkout': date_checkout, }
    if rooms_number:
        querystring['rooms_number'] = rooms_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_hotels(date_checkout: str, sort_order: str, location_id: str, date_checkin: str, amenities_ids: str='FINTRNT,FBRKFST', page_number: int=0, rooms_number: int=1, star_rating_ids: str='3.0,3.5,4.0,4.5,5.0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available hotels by the filter. Indicate the `location_id` -> use `Search locations`, check-in and check-out date"
    date_checkout: Checkout date
        location_id: Location id, use `Search locations` api point
        date_checkin: Checkin date
        amenities_ids: Amenities
        page_number: Number of page
        rooms_number: Rooms number
        star_rating_ids: Hotel star ratings
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v1/hotels/search"
    querystring = {'date_checkout': date_checkout, 'sort_order': sort_order, 'location_id': location_id, 'date_checkin': date_checkin, }
    if amenities_ids:
        querystring['amenities_ids'] = amenities_ids
    if page_number:
        querystring['page_number'] = page_number
    if rooms_number:
        querystring['rooms_number'] = rooms_number
    if star_rating_ids:
        querystring['star_rating_ids'] = star_rating_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_filter_amenities(resume_key: str=None, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads an Amenity list filtered "
    resume_key: Resume results from given ID.
        limit: Limits the number of results from the response.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/downloadFilterAmenities"
    querystring = {}
    if resume_key:
        querystring['resume_key'] = resume_key
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_cities_clusters(limit: int=100, resume_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads a list of Hotel cities clusters"
    limit: Limits the number of results from the response.
        resume_key: Resume results from given ID.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/downloadCitiesClusters"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if resume_key:
        querystring['resume_key'] = resume_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_details(hotel_id: str, airport_limit: int=None, check_in: str=None, promo: bool=None, photos: bool=None, videos: bool=None, guest_score_breakdown: bool=None, reviews: bool=None, city_limit: int=None, sid: str='iSiX639', important_info: bool=None, recent: bool=None, poi_limit: int=None, plugins: bool=None, image_size: str=None, id_lookup: bool=None, check_out: str=None, currency: str=None, nearby: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides information on the specified hotel"
    hotel_id: The PPN Hotel ID identifying the desired property.
        airport_limit: airport_limit
        check_in: Check In Date (YYYY-MM-DD or MM/DD/YYYY)
        promo: Toggle the hotels promo data on and off. By default, promo data is excluded in the response. To include promo data in the response, set promo to 1.
        photos: Toggles photo data. By default, hotel photo data is excluded in the response. Valid Options: 0 = Off, 1 = On.
        videos: Toggle videos. Valid Options: True or False
        guest_score_breakdown: Toggle guest score breakdown on and off. Valid Options: True or False
        reviews: Toggle hotel review data. By default, review data is excluded in the response. Valid Options: 0 = False, 1 = On.
        city_limit: city_limit
        sid: Session ID. Random string
        important_info: Toggles important info. Important info is extra details regarding the specified hotel. By default, important info is excluded in the response. Valid Options: 0 = Off, 1 = On.
        recent: Toggle recent sales. Valid Options: True, False
        poi_limit: poi_limit
        plugins: Provides extra information regarding plugins for the specified hotel. To include specific plugin information, include the ID of the plugin as a comma separated value to the plugins parameter.
        image_size: The size of the image returned. Valid Options: small (60px), medium(300 to 312px) or large(500 to 800px)
        id_lookup: Allows non-PPN Hotel IDs to be searched in the database. Valid Options: True or False
        check_out: Check In Date (YYYY-MM-DD or MM/DD/YYYY)
        currency: Requested currency for the results. ISO 4217 format.
        nearby: Toggles nearby data. Nearby data is extra information regarding the specified hotel s location. Nearby data can include city_data, airport_data, and poi_category_data. By default, nearby data is excluded from the response. To include nearby data, set nearby to 1. To include city_data within nearby data, include a city_limit parameter to a value greater than 1. To include airport_data within nearby data, include a airport_limit parameter to a value greater than 1. To include poi_category_data within nearby data, include a poi_limit parameter to a value greater than 1.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/details"
    querystring = {'hotel_id': hotel_id, }
    if airport_limit:
        querystring['airport_limit'] = airport_limit
    if check_in:
        querystring['check_in'] = check_in
    if promo:
        querystring['promo'] = promo
    if photos:
        querystring['photos'] = photos
    if videos:
        querystring['videos'] = videos
    if guest_score_breakdown:
        querystring['guest_score_breakdown'] = guest_score_breakdown
    if reviews:
        querystring['reviews'] = reviews
    if city_limit:
        querystring['city_limit'] = city_limit
    if sid:
        querystring['sid'] = sid
    if important_info:
        querystring['important_info'] = important_info
    if recent:
        querystring['recent'] = recent
    if poi_limit:
        querystring['poi_limit'] = poi_limit
    if plugins:
        querystring['plugins'] = plugins
    if image_size:
        querystring['image_size'] = image_size
    if id_lookup:
        querystring['id_lookup'] = id_lookup
    if check_out:
        querystring['check_out'] = check_out
    if currency:
        querystring['currency'] = currency
    if nearby:
        querystring['nearby'] = nearby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_car_rentals(location_pickup: str, location_return: str, date_time_pickup: str, date_time_return: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search car rentals by filter. Indicate the `location_id` -> use `Search locations` api point"
    location_pickup: Location pickup code or id. Ex: JFK or 1365100023, use `Search locations` api point
        location_return: Location return code or id
        date_time_pickup: Pickup date and time
        date_time_return: Return date and time
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v1/cars-rentals/search"
    querystring = {'location_pickup': location_pickup, 'location_return': location_return, 'date_time_pickup': date_time_pickup, 'date_time_return': date_time_return, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_flights(date_departure: str, location_departure: str, class_type: str, sort_order: str, itinerary_type: str, location_arrival: str, price_max: int=20000, price_min: int=100, number_of_stops: int=1, date_departure_return: str='2023-10-19', number_of_passengers: int=1, duration_max: int=2051, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search flights. Set type: `ONE_WAY` or `ROUND_TRIP`. Set location_departure and location_arrival, use `/flights/locations` api point. You can filter out tickets by price, max duration and number of stops"
    date_departure: Departure date
        location_departure: Departure location code. Use `Search locations` api point
        location_arrival: Arrival location  code
        price_max: Price max
        price_min: Price min
        number_of_stops: Number of stops. 0 - is direct flight
        date_departure_return: Departure date back
        number_of_passengers: Number of passengers
        duration_max: Duration max. Minutes
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v1/flights/search"
    querystring = {'date_departure': date_departure, 'location_departure': location_departure, 'class_type': class_type, 'sort_order': sort_order, 'itinerary_type': itinerary_type, 'location_arrival': location_arrival, }
    if price_max:
        querystring['price_max'] = price_max
    if price_min:
        querystring['price_min'] = price_min
    if number_of_stops:
        querystring['number_of_stops'] = number_of_stops
    if date_departure_return:
        querystring['date_departure_return'] = date_departure_return
    if number_of_passengers:
        querystring['number_of_passengers'] = number_of_passengers
    if duration_max:
        querystring['duration_max'] = duration_max
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_round_trip(sid: str, adults: int, departure_date: str, airline_filter: str=None, destination_airport_code: str='JFK,YWG', destination_city_id: str=None, cabin_class: str=None, origin_city_id: str=None, page: int=None, origin_airport_code: str='YWG,JFK', convert_currency: str=None, children: int=None, number_of_itineraries: int=None, currency: str=None, results_per_page: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a contract for a flight round trip search through the getFlightRoundTrip endpoint"
    sid: Session ID. Random string
        adults: Number of adults
        departure_date: Departure date
        airline_filter: 2 Letter code used to specify which airline that has been used.
        destination_airport_code: Airport code
        destination_city_id: City id
        cabin_class: economy premium business first
        origin_city_id: City id
        page: How many pages the results are spread over. Used in conjunction with results per page.
        origin_airport_code: Airport code
        convert_currency: Requested currency for the results. ISO 4217 format.
        children: Number of children
        number_of_itineraries: Number of itineraries to retrieve
        currency: Requested currency for the results. ISO 4217 format.
        results_per_page: Number of results per page. Used in conjunction with page.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/flight/roundTrip"
    querystring = {'sid': sid, 'adults': adults, 'departure_date': departure_date, }
    if airline_filter:
        querystring['airline_filter'] = airline_filter
    if destination_airport_code:
        querystring['destination_airport_code'] = destination_airport_code
    if destination_city_id:
        querystring['destination_city_id'] = destination_city_id
    if cabin_class:
        querystring['cabin_class'] = cabin_class
    if origin_city_id:
        querystring['origin_city_id'] = origin_city_id
    if page:
        querystring['page'] = page
    if origin_airport_code:
        querystring['origin_airport_code'] = origin_airport_code
    if convert_currency:
        querystring['convert_currency'] = convert_currency
    if children:
        querystring['children'] = children
    if number_of_itineraries:
        querystring['number_of_itineraries'] = number_of_itineraries
    if currency:
        querystring['currency'] = currency
    if results_per_page:
        querystring['results_per_page'] = results_per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_airports(resume_key: str=None, limit: int=500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads a list of airports with IATA codes for Flight search"
    resume_key: Resume results from given ID.
        limit: Limits the number of results from the response.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/flight/downloadAirports"
    querystring = {}
    if resume_key:
        querystring['resume_key'] = resume_key
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def seat_map(ppn_bundle: str, sid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the seat map of all flights in a contract bundle through the getFlightSeatMap endpoint"
    ppn_bundle: The ppn_bundle for the seat map. Can be retrieved from the ppn_seat_bundle of FlightContract, or FlightLookUp.
        sid: Session ID. Random string ex.: j10k11l12m13n14
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/flight/seatMap"
    querystring = {'ppn_bundle': ppn_bundle, 'sid': sid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(string: str, regions: bool=None, spellcheck: bool=None, airports: bool=None, cities: bool=None, longitude: str=None, pois: bool=None, latitude: str=None, hotels: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets airport and city ids for the air product related to words in passed string through the getAutoComplete endpoint"
    string: Airport or City being searched
        regions: Include regions in search results
        spellcheck: If the spell check is strict.
        airports: Include airports in search results
        cities: Include cities in search results
        longitude: Search for property availability around a specific longitude coordinate.
        pois: Include pois in search results
        latitude: Search for property availability around a specific latitude coordinate.
        hotels: Include hotels in search results
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/flight/autoComplete"
    querystring = {'string': string, }
    if regions:
        querystring['regions'] = regions
    if spellcheck:
        querystring['spellcheck'] = spellcheck
    if airports:
        querystring['airports'] = airports
    if cities:
        querystring['cities'] = cities
    if longitude:
        querystring['longitude'] = longitude
    if pois:
        querystring['pois'] = pois
    if latitude:
        querystring['latitude'] = latitude
    if hotels:
        querystring['hotels'] = hotels
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def contract(sid: str, ppn_bundle: str=None, convert_currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the contract for the PPN bundle provided by a flight return, departure, or combined (round trip/multi-city) through the getFlightContract endpoint"
    sid: Session ID. Random string ex.: j10k11l12m13n14
        ppn_bundle: The ppn_bundle for the seat map. Can be retrieved from the ppn_seat_bundle of Flight Contract, or LookUp
        convert_currency: Requested currency for the results. ISO 4217 format.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/flight/contract"
    querystring = {'sid': sid, }
    if ppn_bundle:
        querystring['ppn_bundle'] = ppn_bundle
    if convert_currency:
        querystring['convert_currency'] = convert_currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_return_flights(sid: str, ppn_bundle: str, convert_currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets all the return flights for a given contract/return bundle from FlightDepartures through the getFlightReturns endpoint"
    sid: Session ID. Random string ex.: j10k11l12m13n14
        ppn_bundle: ppn_bundle is a unique ID that ppn uses to identify a specific rate. This is the return bundle that is provided by FlightDepartures only.
        convert_currency: Requested currency for the results. ISO 4217 format.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/flight/returns"
    querystring = {'sid': sid, 'ppn_bundle': ppn_bundle, }
    if convert_currency:
        querystring['convert_currency'] = convert_currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_departures_one_way(adults: int, sid: str, departure_date: str, origin_airport_code: str='YWG', destination_city_id: str=None, number_of_itineraries: int=None, results_per_page: int=None, destination_airport_code: str='JFK', convert_currency: str=None, cabin_class: str=None, children: int=None, currency: str=None, page: int=None, origin_city_id: str=None, airline_filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a set of flights for departure (one way flight search or return/ multi-city flight search) through the getFlightDepartures endpoint"
    adults: Number of adults
        sid: Session ID. Random string
        departure_date: Departure date
        origin_airport_code: Airport code
        destination_city_id: City id
        number_of_itineraries: Number of itineraries to retrieve
        results_per_page: Number of results per page. Used in conjunction with page.
        destination_airport_code: Airport code
        convert_currency: Requested currency for the results. ISO 4217 format.
        cabin_class: economy premium business first
        children: Number of children
        currency: Requested currency for the results. ISO 4217 format.
        page: How many pages the results are spread over. Used in conjunction with results per page.
        origin_city_id: City id
        airline_filter: 2 Letter code used to specify which airline that has been used.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/flight/departures"
    querystring = {'adults': adults, 'sid': sid, 'departure_date': departure_date, }
    if origin_airport_code:
        querystring['origin_airport_code'] = origin_airport_code
    if destination_city_id:
        querystring['destination_city_id'] = destination_city_id
    if number_of_itineraries:
        querystring['number_of_itineraries'] = number_of_itineraries
    if results_per_page:
        querystring['results_per_page'] = results_per_page
    if destination_airport_code:
        querystring['destination_airport_code'] = destination_airport_code
    if convert_currency:
        querystring['convert_currency'] = convert_currency
    if cabin_class:
        querystring['cabin_class'] = cabin_class
    if children:
        querystring['children'] = children
    if currency:
        querystring['currency'] = currency
    if page:
        querystring['page'] = page
    if origin_city_id:
        querystring['origin_city_id'] = origin_city_id
    if airline_filter:
        querystring['airline_filter'] = airline_filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_results_request(dropoff_time: str, pickup_date: str, dropoff_date: str, pickup_time: str, currency: str=None, pickup_lat_long: str=None, pickup_city_string: str=None, result_offset: int=None, strict_city: str=None, pickup_city_id: str=None, dropoff_lat_long: str=None, result_limit: int=None, dropoff_city_string: str='New York', dropoff_airport_code: str=None, car_change_bundle: str=None, dropoff_counters: str=None, dropoff_city_id: str='800049480', pickup_counters: str=None, sort: str=None, pickup_airport_code: str='JFK', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The getResultsRequest endpoint returns car availability based on the provided search criteria"
    dropoff_time: Car Drop Off Time (HH:mm:ss)
        pickup_date: Car Pick Up Date (YYYY-MM-DD or MM/DD/YYYY)
        dropoff_date: Car Drop Off Date (YYYY-MM-DD or MM/DD/YYYY).
        pickup_time: Car Pick Up Time (HH:mm:ss)
        currency: Requested currency for the results. ISO 4217 format.
        pickup_lat_long: Pickup location by latitude and longitude. Accepts a comma separated latitude and longitude values.
        pickup_city_string: Pickup city name.
        result_offset: Used with limit to only retrieve a subset of all results at a time. Determines the number of properties to skip (starting at 0) before returning results.
        strict_city: Filter strictly by city ID (pickup_city_id or dropoff_city_id). Pass strict_city=true or strict_city=1 to enable.
        pickup_city_id: Pickup city ID. Or filters by pickup_city_id when strict_city=1 is passed.
        dropoff_lat_long: dropoff location by latitude and longitude. Accepts a comma separated latitude and longitude values.
        result_limit: Used with offset to only retrieve a subset of all results at a time. Determines the number of properties to return with each call.
        dropoff_city_string: Dropoff city name.
        dropoff_airport_code: Accepts a 3-character IATA airport code.
        car_change_bundle: Search using the car change bundle data. This bundle value is provided in the Change endpoint response when you request a booking change. With  this bundle data, you can search results for the changed search criteria.
        dropoff_counters: Accepts an array of one or more keys as a 2-character car company code and the values as the counter ocation code.
        dropoff_city_id: dropoff city ID. Or filters by dropoff_city_id when strict_city=1 is passed.
        pickup_counters: Accepts an array of one or more keys as a 2-character car company code and the values as the counter location code.
        sort: Method of sorting the results. Valid options: PRICE_HIGH, PRICE_DESC, PARTNER_HIGH, and PARTNER_DESC.
        pickup_airport_code: Accepts a 3-character IATA airport code.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/cars/resultsRequest"
    querystring = {'dropoff_time': dropoff_time, 'pickup_date': pickup_date, 'dropoff_date': dropoff_date, 'pickup_time': pickup_time, }
    if currency:
        querystring['currency'] = currency
    if pickup_lat_long:
        querystring['pickup_lat_long'] = pickup_lat_long
    if pickup_city_string:
        querystring['pickup_city_string'] = pickup_city_string
    if result_offset:
        querystring['result_offset'] = result_offset
    if strict_city:
        querystring['strict_city'] = strict_city
    if pickup_city_id:
        querystring['pickup_city_id'] = pickup_city_id
    if dropoff_lat_long:
        querystring['dropoff_lat_long'] = dropoff_lat_long
    if result_limit:
        querystring['result_limit'] = result_limit
    if dropoff_city_string:
        querystring['dropoff_city_string'] = dropoff_city_string
    if dropoff_airport_code:
        querystring['dropoff_airport_code'] = dropoff_airport_code
    if car_change_bundle:
        querystring['car_change_bundle'] = car_change_bundle
    if dropoff_counters:
        querystring['dropoff_counters'] = dropoff_counters
    if dropoff_city_id:
        querystring['dropoff_city_id'] = dropoff_city_id
    if pickup_counters:
        querystring['pickup_counters'] = pickup_counters
    if sort:
        querystring['sort'] = sort
    if pickup_airport_code:
        querystring['pickup_airport_code'] = pickup_airport_code
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_cities(limit: int=500, resume_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads a list of cities"
    limit: Limits the number of results from the response.
        resume_key: Resume results from given ID.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/cars/downloadCities"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if resume_key:
        querystring['resume_key'] = resume_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_locations(limit: int=500, resume_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads a list of Locations"
    limit: Limits the number of results from the response.
        resume_key: Resume results from given ID.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/cars/downloadLocations"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if resume_key:
        querystring['resume_key'] = resume_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_flights(location_arrival: str, sort_order: str, date_departure: str, itinerary_type: str, class_type: str, location_departure: str, number_of_stops: int=1, price_max: int=20000, number_of_passengers: int=1, duration_max: int=2051, price_min: int=100, date_departure_return: str='2023-10-19', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search flights. Set type: `ONE_WAY` or `ROUND_TRIP`. Set location_departure and location_arrival, use `/flights/locations` api point. You can filter out tickets by price, max duration and number of stops"
    location_arrival: Arrival location  code
        date_departure: Departure date
        location_departure: Departure location code. Use `Search locations` api point
        number_of_stops: Number of stops. 0 - is direct flight
        price_max: Price max
        number_of_passengers: Number of passengers
        duration_max: Duration max. Minutes
        price_min: Price min
        date_departure_return: Departure date back
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/community/v1/flights/search"
    querystring = {'location_arrival': location_arrival, 'sort_order': sort_order, 'date_departure': date_departure, 'itinerary_type': itinerary_type, 'class_type': class_type, 'location_departure': location_departure, }
    if number_of_stops:
        querystring['number_of_stops'] = number_of_stops
    if price_max:
        querystring['price_max'] = price_max
    if number_of_passengers:
        querystring['number_of_passengers'] = number_of_passengers
    if duration_max:
        querystring['duration_max'] = duration_max
    if price_min:
        querystring['price_min'] = price_min
    if date_departure_return:
        querystring['date_departure_return'] = date_departure_return
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_airports(resume_key: str=None, limit: int=500, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads a list of airports with IATA codes for Flight search"
    resume_key: Resume results from given ID.
        limit: Limits the number of results from the response.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/cars/downloadAirports"
    querystring = {}
    if resume_key:
        querystring['resume_key'] = resume_key
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(string: str, get_airports_in_cities: bool=None, get_cities: bool=None, max_results: int=None, get_airports: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets airport and city ids for the air product related to words in passed string."
    string: Search keyword. Airport or City being searched.
        get_airports_in_cities: If City is used as string include airports in that city. 
        get_cities: Include cities in search results. Valid Options: True or False.
        max_results: Number passed is the maximum number of results returned.
        get_airports: Include airports in search results. Valid Options: True or False.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/cars/autoComplete"
    querystring = {'string': string, }
    if get_airports_in_cities:
        querystring['get_airports_in_cities'] = get_airports_in_cities
    if get_cities:
        querystring['get_cities'] = get_cities
    if max_results:
        querystring['max_results'] = max_results
    if get_airports:
        querystring['get_airports'] = get_airports
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_photos(hotel_ids: str, image_size: str='medium', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns a list of photos per hotel"
    hotel_ids: Comma separated string of PPN hotel ids (Semi Opaque Only)
        image_size: The size of the image returned. Valid Options: small (60px), medium(300 to 312px) or large(500 to 800px)
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/photos"
    querystring = {'hotel_ids': hotel_ids, }
    if image_size:
        querystring['image_size'] = image_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_express_multi_contract(children: int=None, rooms: int=None, output_version: int=3, rate_identifier: bool=None, language: str='fr-FR', adults: int=None, include_prepaid_fee_rates: bool=None, check_out: str=None, check_in: str=None, country_code: str='BR', ppn_bundle: str=None, hotel_id: str=None, sid: str='iSiX639', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides hotel and multiple rates information per hotel using the getExpress.MultiContract endpoint."
    children: The total number of child occupants for all rooms requested. Used with adults parameter to determine occupancy. Example: Two rooms, each with one adult and one child occupants, adults=2 and children=2
        rooms: Number of rooms required for all occupants
        output_version: Enum: 1 2 3 4
        rate_identifier: A toggle to show if rate identifier is being passed. Valid Options: 0 = false, 1 = true. Rate is a string that is set for each hotel and holds all the information regarding the rate that we send to priceline.
        language: Language code: en-US, es-ES, fr-FR, pt-BR
        adults: The total number of adult occupants for all rooms requested. Used with children parameter to determine occupancy. Example: Two rooms, each with one adult and one child occupants, adults=2 and children=2
        include_prepaid_fee_rates: Request for rates that have mandatory resort fees that can be paid at time of booking
        check_out: Check In Date (YYYY-MM-DD or MM/DD/YYYY)
        check_in: Check In Date (YYYY-MM-DD or MM/DD/YYYY)
        country_code: Pass the user s country to see rates with regional pricing. This is a two character ISO Alpha-2 country code.
        ppn_bundle: ppn_bundle is a unique ID that ppn uses to identify a specific rate
        hotel_id: Single PPN hotel id (Semi Opaque Only)
        sid: Session ID. Random string
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/expressMultiContract"
    querystring = {}
    if children:
        querystring['children'] = children
    if rooms:
        querystring['rooms'] = rooms
    if output_version:
        querystring['output_version'] = output_version
    if rate_identifier:
        querystring['rate_identifier'] = rate_identifier
    if language:
        querystring['language'] = language
    if adults:
        querystring['adults'] = adults
    if include_prepaid_fee_rates:
        querystring['include_prepaid_fee_rates'] = include_prepaid_fee_rates
    if check_out:
        querystring['check_out'] = check_out
    if check_in:
        querystring['check_in'] = check_in
    if country_code:
        querystring['country_code'] = country_code
    if ppn_bundle:
        querystring['ppn_bundle'] = ppn_bundle
    if hotel_id:
        querystring['hotel_id'] = hotel_id
    if sid:
        querystring['sid'] = sid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_property_types(limit: int=100, resume_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads Property Types list "
    limit: Limits the number of results from the response.
        resume_key: Resume results from given ID.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/downloadPropertyTypes"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if resume_key:
        querystring['resume_key'] = resume_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_chains(resume_key: str=None, limit: int=100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads a list of Hotel chains"
    resume_key: Resume results from given ID.
        limit: Limits the number of results from the response.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/downloadChains"
    querystring = {}
    if resume_key:
        querystring['resume_key'] = resume_key
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_reviews(hotel_id: str, languages: str='en,fr', order_by: str=None, offset: int=None, limit: int=100, only_verified_guests: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API returns a list of reviews"
    hotel_id: The PPN Hotel ID identifying the desired property.
        languages: Limits the number of results from the response.
        order_by: CSV of sorting order metrics. Valid Options: creation_date, average_rating, or verified_guest followed by .asc or .desc.
        offset: Used with limit to only retrieve a subset of all results at a time. Determines the nuber of properties to skip (starting at 0) before returning results.
        limit: Limits the number of results from the response.
        only_verified_guests: Set on to only include only reviews with verified_guests. A verified guest is a guest that has had a review verified by aaa. Valid Options: 0 = Off, 1 = On.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/reviews"
    querystring = {'hotel_id': hotel_id, }
    if languages:
        querystring['languages'] = languages
    if order_by:
        querystring['order_by'] = order_by
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if only_verified_guests:
        querystring['only_verified_guests'] = only_verified_guests
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_areas(limit: int=100, resume_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads an Area list"
    limit: Limits the number of results from the response.
        resume_key: Resume results from given ID.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/downloadAreas"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if resume_key:
        querystring['resume_key'] = resume_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_states(limit: int=100, resume_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads a list of Satets "
    limit: Limits the number of results from the response.
        resume_key: Resume results from given ID.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/downloadStates"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if resume_key:
        querystring['resume_key'] = resume_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def express_contract(sid: str='iSiX639', rate_identifier: bool=None, country_code: str='BR', output_version: int=3, language: str='fr-FR', ppn_bundle: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Provides the hotel inventory and corresponding rates for Express (cache) or Closed User Group (live) "
    sid: Session ID. Random string
        rate_identifier: A toggle to show if rate identifier is being passed. Valid Options: 0 = false, 1 = true. Rate is a string that is set for each hotel and holds all the information regarding the rate that we send to priceline.
        country_code: Pass the user s country to see rates with regional pricing. This is a two character ISO Alpha-2 country code.
        output_version: Enum: 1 2 3 4
        language: Language code: en-US, es-ES, fr-FR, pt-BR
        ppn_bundle: ppn_bundle is a unique ID that ppn uses to identify a specific rate
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/expressContract"
    querystring = {}
    if sid:
        querystring['sid'] = sid
    if rate_identifier:
        querystring['rate_identifier'] = rate_identifier
    if country_code:
        querystring['country_code'] = country_code
    if output_version:
        querystring['output_version'] = output_version
    if language:
        querystring['language'] = language
    if ppn_bundle:
        querystring['ppn_bundle'] = ppn_bundle
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downalods_hotels(active_mer: str=None, latitude_range_end: str=None, active_smop: str=None, active_vmer: str=None, state_code: str=None, longitude_range_end: str=None, active_bkg: str=None, latitude: str=None, hotelid_ppn: str=None, longitude: str=None, property_type_ids: str=None, cityid_ppn: str=None, hotel_address: str=None, resume_key: str=None, language: str='fr-FR', limit: int=100, active_agd: str=None, country_code: str=None, changes_since: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downalods a list of Hotels"
    active_mer: Show hotels with Priceline rates.
        latitude_range_end: Requires latitude to have value.
        active_smop: Show hotels with semi opaque rates.
        active_vmer: Show hotels with vacation merchant rates.
        state_code: Filter by the state code of the hotel.
        longitude_range_end: Requires longitude to have value.
        active_bkg: Show hotels with Booking rates.
        latitude: Filter by latitude of the hotel.
        hotelid_ppn: Filter by PPN hotel ID.
        longitude: Requires longitude to have value.
        property_type_ids: Filter by property type ids. See the Property Type Filter Guide for more detail.
        cityid_ppn: Filter by PPN city ID.
        hotel_address: Filter by address of hotel.
        resume_key: Resume results from given ID.
        language: Language code: en-US, es-ES, fr-FR, pt-BR
        limit: Limits the number of results from the response.
        active_agd: Show hotels with Agoda rates.
        country_code: Filter by the country code of the hotel.
        changes_since: Date/time to filter the hotels that have been updated on or after this date. This will discover the last_changed_date of hotels in inventory (inclusive of the selected date). Date should be in a valid ISO 8601: https://en.wikipedia.org/wiki/ISO_8601 (YYYY-MM-DDThh:mm:ss{UTC_Offset}) format.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/downloadHotels"
    querystring = {}
    if active_mer:
        querystring['active_mer'] = active_mer
    if latitude_range_end:
        querystring['latitude_range_end'] = latitude_range_end
    if active_smop:
        querystring['active_smop'] = active_smop
    if active_vmer:
        querystring['active_vmer'] = active_vmer
    if state_code:
        querystring['state_code'] = state_code
    if longitude_range_end:
        querystring['longitude_range_end'] = longitude_range_end
    if active_bkg:
        querystring['active_bkg'] = active_bkg
    if latitude:
        querystring['latitude'] = latitude
    if hotelid_ppn:
        querystring['hotelid_ppn'] = hotelid_ppn
    if longitude:
        querystring['longitude'] = longitude
    if property_type_ids:
        querystring['property_type_ids'] = property_type_ids
    if cityid_ppn:
        querystring['cityid_ppn'] = cityid_ppn
    if hotel_address:
        querystring['hotel_address'] = hotel_address
    if resume_key:
        querystring['resume_key'] = resume_key
    if language:
        querystring['language'] = language
    if limit:
        querystring['limit'] = limit
    if active_agd:
        querystring['active_agd'] = active_agd
    if country_code:
        querystring['country_code'] = country_code
    if changes_since:
        querystring['changes_since'] = changes_since
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def download_amenities(limit: int=100, language: str='fr-FR', resume_key: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads a list of Amenities"
    limit: Limits the number of results from the response.
        language: Language code: en-US, es-ES, fr-FR, pt-BR
        resume_key: Resume results from given ID.
        
    """
    url = f"https://priceline-com-provider.p.rapidapi.com/v2/hotels/downloadAmenities"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    if language:
        querystring['language'] = language
    if resume_key:
        querystring['resume_key'] = resume_key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "priceline-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

