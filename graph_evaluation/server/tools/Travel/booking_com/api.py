import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_cars_locations(locale: str, name: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search cars locations by name"
    name: Name
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/car-rental/locations"
    querystring = {'locale': locale, 'name': name, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_car_rental(currency: str, drop_off_latitude: int, sort_by: str, drop_off_datetime: str, from_country: str, pick_up_latitude: int, locale: str, pick_up_datetime: str, drop_off_longitude: int, pick_up_longitude: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search car rental. Browse our inventory of hundreds of cheap car rentals. Get the best rental car deals on your next trip"
    drop_off_latitude: Drop off latitude
        drop_off_datetime: Drop off datetime
        pick_up_latitude: Pick up latitude
        pick_up_datetime: Pick up datetime
        drop_off_longitude: Drop off longitude
        pick_up_longitude: Pick up longitude
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/car-rental/search"
    querystring = {'currency': currency, 'drop_off_latitude': drop_off_latitude, 'sort_by': sort_by, 'drop_off_datetime': drop_off_datetime, 'from_country': from_country, 'pick_up_latitude': pick_up_latitude, 'locale': locale, 'pick_up_datetime': pick_up_datetime, 'drop_off_longitude': drop_off_longitude, 'pick_up_longitude': pick_up_longitude, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vehicle_supplier_details(from_country: str, locale: str, vehicle_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Vehicle supplier details by vehicle_id"
    vehicle_id: Vehicle id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/car-rental/supplier/details"
    querystring = {'from_country': from_country, 'locale': locale, 'vehicle_id': vehicle_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def important_information(locale: str, pick_up_location_id: int, drop_off_location_id: int, drop_off_datetime: str, from_country: str, pick_up_datetime: str, driver_age: int, vehicle_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Important information of vehicles"
    pick_up_location_id: Pick up location id
        drop_off_location_id: Drop off location id
        drop_off_datetime: Drop off datetime
        pick_up_datetime: Pick up datetime
        driver_age: Driver age
        vehicle_id: Vehicle id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/car-rental/important-info"
    querystring = {'locale': locale, 'pick_up_location_id': pick_up_location_id, 'drop_off_location_id': drop_off_location_id, 'drop_off_datetime': drop_off_datetime, 'from_country': from_country, 'pick_up_datetime': pick_up_datetime, 'driver_age': driver_age, 'vehicle_id': vehicle_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange_rates(currency: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get currency exchange rates"
    
    """
    url = f"https://booking-com.p.rapidapi.com/v1/metadata/exchange-rates"
    querystring = {'currency': currency, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def car_rental_details(vehicle_id: int, currency: str, from_country: str, locale: str, search_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Car rental details by vehicle_id. First, call `Search car rental`"
    vehicle_id: Vehicle id
        search_key: Search key
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/car-rental/detail"
    querystring = {'vehicle_id': vehicle_id, 'currency': currency, 'from_country': from_country, 'locale': locale, 'search_key': search_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rental_terms(from_country: str, pick_up_location_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rental terms and conditions by pick up location_id"
    pick_up_location_id: Pick up location id. Use `Vehicle supplier details`
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/car-rental/terms-conditions"
    querystring = {'from_country': from_country, 'pick_up_location_id': pick_up_location_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rental_extras(pick_up_datetime: str, currency: str, from_country: str, locale: str, price: int, drop_off_datetime: str, driver_age: int, vehicle_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get additional services when renting"
    pick_up_datetime: Pick up datetime
        price: Price
        drop_off_datetime: Drop off datetime
        driver_age: Driver age
        vehicle_id: Vehicle id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/car-rental/extras"
    querystring = {'pick_up_datetime': pick_up_datetime, 'currency': currency, 'from_country': from_country, 'locale': locale, 'price': price, 'drop_off_datetime': drop_off_datetime, 'driver_age': driver_age, 'vehicle_id': vehicle_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_metadata_of_the_hotel(locale: str, hotel_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviews metadata of the hotel by hotel_id"
    hotel_id: Hotel id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/reviews-filter-metadata"
    querystring = {'locale': locale, 'hotel_id': hotel_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nearby_cities(latitude: int, longitude: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of nearby cities by coordinates"
    latitude: Latitude
        longitude: Longitude
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/nearby-cities"
    querystring = {'latitude': latitude, 'longitude': longitude, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def questions_about_the_hotel(hotel_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get questions and answers of hotel staff by hotel_id"
    hotel_id: Hotel id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/questions"
    querystring = {'hotel_id': hotel_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_on_the_map(hotel_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get nearby landmarks and geo info of the hotel by hotel_id"
    hotel_id: Hotel id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/map-markers"
    querystring = {'hotel_id': hotel_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def room_list_of_the_hotel(hotel_id: int, currency: str, checkout_date: str, locale: str, checkin_date: str, adults_number_by_rooms: str, units: str, children_ages: str='5,0,9', children_number_by_rooms: str='2,1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get room list of the hotel by hotel_id. Room prices, photos and full room descriptions"
    hotel_id: Hotel id
        checkout_date: Checkout date
        checkin_date: Checkin date
        adults_number_by_rooms: The number of adults in each room. Specify the number of adults separated by commas, for example: `3,1` Means that the first room will accommodate three adults, and the second room will accommodate 1 adult. If you are looking for just one room for two adults, please specify `2`
        children_ages: The age of every child. If children will be staying in the room, please indicate their ages separated by commas. 0-less than a year
        children_number_by_rooms: The number of children in each room. Specify the number of children separated by commas, for example: `3,1` Means that the first room will accommodate three children, and the second room will accommodate 1 children
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/room-list"
    querystring = {'hotel_id': hotel_id, 'currency': currency, 'checkout_date': checkout_date, 'locale': locale, 'checkin_date': checkin_date, 'adults_number_by_rooms': adults_number_by_rooms, 'units': units, }
    if children_ages:
        querystring['children_ages'] = children_ages
    if children_number_by_rooms:
        querystring['children_number_by_rooms'] = children_number_by_rooms
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_about_the_supplier_of_vehicles(locale: str, location_id: int, from_country: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reviews about the supplier of vehicles by location_id"
    location_id: Location id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/car-rental/supplier/reviews"
    querystring = {'locale': locale, 'location_id': location_id, 'from_country': from_country, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def location_highlights_of_the_hotel(hotel_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get location highlights of the hotel by hotel_id. Ex.: metro, rail"
    hotel_id: Hotel id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/location-highlights"
    querystring = {'hotel_id': hotel_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def description_of_the_hotel(hotel_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get description of the hotel by hotel_id"
    hotel_id: Hotel id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/description"
    querystring = {'hotel_id': hotel_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def review_scores_of_the_hotel(hotel_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get review scores of the hotel by hotel_id"
    hotel_id: Hotel id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/review-scores"
    querystring = {'hotel_id': hotel_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_children_policy(locale: str, hotel_id: int, children_age: int=5, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hotel children policy by hotel_id"
    hotel_id: Hotel id
        children_age: Children age
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/children-policies"
    querystring = {'locale': locale, 'hotel_id': hotel_id, }
    if children_age:
        querystring['children_age'] = children_age
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_hotels_by_coordinates(units: str, room_number: int, longitude: int, latitude: int, filter_by_currency: str, order_by: str, locale: str, checkout_date: str, adults_number: int, checkin_date: str, children_ages: str='5,0', include_adjacency: bool=None, children_number: int=2, page_number: int=0, categories_filter_ids: str='class::2,class::4,free_cancellation::1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available hotels by coordinates latitude and longitude"
    room_number: Number of rooms
        longitude: Longitude
        latitude: Latitude
        checkout_date: Checkout date
        adults_number: Number of adults
        checkin_date: Checkin date
        children_ages: The age of every child. If children will be staying in the room, please indicate their ages separated by commas. 0-less than a year
        include_adjacency: Include nearby places. If there are few hotels in the selected location, nearby locations will be added. You should pay attention to the `primary_count` parameter - it is the number of hotels from the beginning of the array that matches the strict filter.
        children_number: Number of children
        page_number: Page number
        categories_filter_ids: Search filter IDs. For possible filters use `@Filters for search`
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/search-by-coordinates"
    querystring = {'units': units, 'room_number': room_number, 'longitude': longitude, 'latitude': latitude, 'filter_by_currency': filter_by_currency, 'order_by': order_by, 'locale': locale, 'checkout_date': checkout_date, 'adults_number': adults_number, 'checkin_date': checkin_date, }
    if children_ages:
        querystring['children_ages'] = children_ages
    if include_adjacency:
        querystring['include_adjacency'] = include_adjacency
    if children_number:
        querystring['children_number'] = children_number
    if page_number:
        querystring['page_number'] = page_number
    if categories_filter_ids:
        querystring['categories_filter_ids'] = categories_filter_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def policies_of_the_hotel(locale: str, hotel_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get policies of the hotel by hotel_id"
    hotel_id: Hotel id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/policies"
    querystring = {'locale': locale, 'hotel_id': hotel_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def payment_features_of_the_hotel(hotel_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get payment features of the hotel by hotel_id"
    hotel_id: Hotel id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/payment-features"
    querystring = {'hotel_id': hotel_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_of_the_hotel(sort_type: str, hotel_id: int, locale: str, page_number: int=0, language_filter: str='en-gb,de,fr', customer_type: str='solo_traveller,review_category_group_of_friends', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviews of the hotel by hotel_id"
    hotel_id: Hotel id
        page_number: Page number
        language_filter: Filter reviews by language
        customer_type: Customer types
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/reviews"
    querystring = {'sort_type': sort_type, 'hotel_id': hotel_id, 'locale': locale, }
    if page_number:
        querystring['page_number'] = page_number
    if language_filter:
        querystring['language_filter'] = language_filter
    if customer_type:
        querystring['customer_type'] = customer_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_hotels(checkin_date: str, dest_type: str, units: str, checkout_date: str, adults_number: int, order_by: str, dest_id: int, filter_by_currency: str, locale: str, room_number: int, children_number: int=2, children_ages: str='5,0', categories_filter_ids: str='class::2,class::4,free_cancellation::1', page_number: int=0, include_adjacency: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available hotels by the filter. Indicate the `destination_id` and `dest_type` -> use **@Search locations** endpoint, check-in and check-out date, number of adults and children. For possible filters use **@Filters for search** endpoint. Use `page_number` to paginate and navigate through the result pages! <br/> Comparable to: ![Search locations](https://i.imgur.com/PCuK81r.png)"
    checkin_date: Checkin date
        checkout_date: Checkout date
        adults_number: Number of adults
        dest_id: Destination id, use `Search locations` to find a place, field `dest_id` and `dest_type`
        room_number: Number of rooms
        children_number: Number of children
        children_ages: The age of every child. If children will be staying in the room, please indicate their ages separated by commas. 0-less than a year
        categories_filter_ids: Search filter IDs. For possible filters use `@Filters for search`
        page_number: Page number
        include_adjacency: Include nearby places. If there are few hotels in the selected location, nearby locations will be added. You should pay attention to the `primary_count` parameter - it is the number of hotels from the beginning of the array that matches the strict filter.
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/search"
    querystring = {'checkin_date': checkin_date, 'dest_type': dest_type, 'units': units, 'checkout_date': checkout_date, 'adults_number': adults_number, 'order_by': order_by, 'dest_id': dest_id, 'filter_by_currency': filter_by_currency, 'locale': locale, 'room_number': room_number, }
    if children_number:
        querystring['children_number'] = children_number
    if children_ages:
        querystring['children_ages'] = children_ages
    if categories_filter_ids:
        querystring['categories_filter_ids'] = categories_filter_ids
    if page_number:
        querystring['page_number'] = page_number
    if include_adjacency:
        querystring['include_adjacency'] = include_adjacency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tips_of_the_hotel(hotel_id: int, sort_type: str, locale: str, language_filter: str='en-gb,de,fr', page_number: int=0, customer_type: str='solo_traveller,review_category_group_of_friends', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get guest tips of the hotel by hotel_id"
    hotel_id: Hotel id
        language_filter: Filter reviews by language
        page_number: Page number
        customer_type: Customer types
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/tips"
    querystring = {'hotel_id': hotel_id, 'sort_type': sort_type, 'locale': locale, }
    if language_filter:
        querystring['language_filter'] = language_filter
    if page_number:
        querystring['page_number'] = page_number
    if customer_type:
        querystring['customer_type'] = customer_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def facilities_of_the_hotel(hotel_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get facilities of the hotel by hotel_id. Ex.: `Restaurant`, `Room service`, `Breakfast in the room`"
    hotel_id: Hotel id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/facilities"
    querystring = {'hotel_id': hotel_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_locations(name: str, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search locations or hotels by name. Comparable to: ![Search locations](https://i.imgur.com/wLgLE5X.png)"
    name: Name
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/locations"
    querystring = {'name': name, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nearby_places_of_the_hotel(hotel_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get nearby places of the hotel by hotel_id. Ex.: nearby `Airport`, `Shopping Centre`"
    hotel_id: Hotel id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/nearby-places"
    querystring = {'hotel_id': hotel_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def photos_of_the_hotel(hotel_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all photos of the hotel by hotel_id"
    hotel_id: Hotel id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/photos"
    querystring = {'hotel_id': hotel_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filters_for_search(locale: str, dest_id: int, checkin_date: str, filter_by_currency: str, checkout_date: str, dest_type: str, units: str, order_by: str, adults_number: int, room_number: int, page_number: int=0, children_number: int=2, include_adjacency: bool=None, categories_filter_ids: str='class::2,class::4,free_cancellation::1', children_ages: str='5,0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of available filters for hotel search"
    dest_id: Destination id, use `Search locations` to find a place, field `dest_id` and `dest_type`
        checkin_date: Checkin date
        checkout_date: Checkout date
        adults_number: Number of adults
        room_number: Number of rooms
        page_number: Page number
        children_number: Number of children
        include_adjacency: Include nearby places. If there are few hotels in the selected location, nearby locations will be added. You should pay attention to the `primary_count` parameter - it is the number of hotels from the beginning of the array that matches the strict filter.
        categories_filter_ids: Search filter IDs. For possible filters use `@Filters for search`
        children_ages: The age of every child. If children will be staying in the room, please indicate their ages separated by commas. 0-less than a year
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/search-filters"
    querystring = {'locale': locale, 'dest_id': dest_id, 'checkin_date': checkin_date, 'filter_by_currency': filter_by_currency, 'checkout_date': checkout_date, 'dest_type': dest_type, 'units': units, 'order_by': order_by, 'adults_number': adults_number, 'room_number': room_number, }
    if page_number:
        querystring['page_number'] = page_number
    if children_number:
        querystring['children_number'] = children_number
    if include_adjacency:
        querystring['include_adjacency'] = include_adjacency
    if categories_filter_ids:
        querystring['categories_filter_ids'] = categories_filter_ids
    if children_ages:
        querystring['children_ages'] = children_ages
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_districts(name: str=None, country: str=None, city_id: str=None, district_id: str=None, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the districts where Booking offers hotels."
    name: District name
        country: Country ID
        city_id: City ID
        district_id: District ID
        page: Page number
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/static/districts"
    querystring = {}
    if name:
        querystring['name'] = name
    if country:
        querystring['country'] = country
    if city_id:
        querystring['city_id'] = city_id
    if district_id:
        querystring['district_id'] = district_id
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_data(hotel_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hotel data by hotel_id"
    hotel_id: Hotel id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/hotels/data"
    querystring = {'hotel_id': hotel_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_payment_types(payment_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of payments. Only payments with {bookable} set to 1, can be used in processBooking. For processBooking, we always support Visa, Mastercard and American Express."
    payment_id: Payment ID
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/static/payment-types"
    querystring = {}
    if payment_id:
        querystring['payment_id'] = payment_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_cities(page: int=0, city_id: str=None, name: str=None, latitude: str=None, longitude: str=None, country: str='it', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of cities where Booking.com offers hotels. location: Latitude and longitude. timezone: Timezone of the city."
    page: Page number
        city_id: City ID
        name: City name
        latitude: Latitude location
        longitude: Longitude location
        country: Country ID
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/static/cities"
    querystring = {}
    if page:
        querystring['page'] = page
    if city_id:
        querystring['city_id'] = city_id
    if name:
        querystring['name'] = name
    if latitude:
        querystring['latitude'] = latitude
    if longitude:
        querystring['longitude'] = longitude
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_hotels(page: int=0, city_id: int=None, slug: str=None, region_id: int=None, country: str=None, exact_class: int=None, hotel_id: int=None, zip_code: str=None, hotel_type_id: int=None, district_id: int=None, name: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This call returns the hotel and room data. List of hotels"
    page: Page number
        city_id: City ID
        slug: Website slug. Ex. `ad/plaza` or `it/castello-delle-serre`
        region_id: Region ID
        country: Country ID
        exact_class: Exact class
        hotel_id: Hotel ID
        zip_code: ZIP Code
        hotel_type_id: Hotel type ID
        district_id: District ID
        name: Hotel name
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/static/hotels"
    querystring = {}
    if page:
        querystring['page'] = page
    if city_id:
        querystring['city_id'] = city_id
    if slug:
        querystring['slug'] = slug
    if region_id:
        querystring['region_id'] = region_id
    if country:
        querystring['country'] = country
    if exact_class:
        querystring['exact_class'] = exact_class
    if hotel_id:
        querystring['hotel_id'] = hotel_id
    if zip_code:
        querystring['zip_code'] = zip_code
    if hotel_type_id:
        querystring['hotel_type_id'] = hotel_type_id
    if district_id:
        querystring['district_id'] = district_id
    if name:
        querystring['name'] = name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_facility_types(facility_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns facility types names and their translations"
    facility_id: Facility ID
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/static/facility-types"
    querystring = {}
    if facility_id:
        querystring['facility_id'] = facility_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_regions(region_id: str=None, page: int=0, name: str=None, country: str=None, region_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the regions where Booking offers hotels."
    region_id: Region IDs
        page: Page number
        name: Region name
        country: Country ID
        region_type: Region types
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/static/regions"
    querystring = {}
    if region_id:
        querystring['region_id'] = region_id
    if page:
        querystring['page'] = page
    if name:
        querystring['name'] = name
    if country:
        querystring['country'] = country
    if region_type:
        querystring['region_type'] = region_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_hotel_types(hotel_type_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns hotel types names and their translations."
    hotel_type_id: Hotel type ID
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/static/hotel-types"
    querystring = {}
    if hotel_type_id:
        querystring['hotel_type_id'] = hotel_type_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_room_facility_types(room_facility_type_id: str=None, facility_type_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns room facility types names"
    room_facility_type_id: Room facility type ID
        facility_type_id: Facility type ID
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/static/room-facility-types"
    querystring = {}
    if room_facility_type_id:
        querystring['room_facility_type_id'] = room_facility_type_id
    if facility_type_id:
        querystring['facility_type_id'] = facility_type_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_hotel_facility_types(hotel_facility_type_id: str=None, facility_type_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns hotel facility types names and their translations"
    hotel_facility_type_id: Hotel facility type ID
        facility_type_id: Facility type ID
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/static/hotel-facility-types"
    querystring = {}
    if hotel_facility_type_id:
        querystring['hotel_facility_type_id'] = hotel_facility_type_id
    if facility_type_id:
        querystring['facility_type_id'] = facility_type_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_room_types(room_type_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns room types names"
    room_type_id: Room type ID
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/static/room-types"
    querystring = {}
    if room_type_id:
        querystring['room_type_id'] = room_type_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_hotel_chains(name: str=None, chain_id: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of hotel chains."
    name: Hotel chain name
        chain_id: Chain ID
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/static/chain-types"
    querystring = {}
    if name:
        querystring['name'] = name
    if chain_id:
        querystring['chain_id'] = chain_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_of_countries(country: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns all the country where booking offers hotels."
    country: Country ID
        
    """
    url = f"https://booking-com.p.rapidapi.com/v1/static/country"
    querystring = {}
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotels_details(hotel_id: int, currency: str, locale: str, checkout_date: str, checkin_date: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hotels Details"
    hotel_id: Hotel id
        checkout_date: Checkout date
        checkin_date: Checkin date
        
    """
    url = f"https://booking-com.p.rapidapi.com/v2/hotels/details"
    querystring = {'hotel_id': hotel_id, 'currency': currency, 'locale': locale, 'checkout_date': checkout_date, 'checkin_date': checkin_date, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotels_search(order_by: str, adults_number: int, checkin_date: str, filter_by_currency: str, dest_id: int, locale: str, checkout_date: str, units: str, room_number: int, dest_type: str, include_adjacency: bool=None, children_number: int=2, page_number: int=0, children_ages: str='5,0', categories_filter_ids: str='class::2,class::4,free_cancellation::1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available hotels by the filter. Indicate the `destination_id` and `dest_type` -> use **@Search locations** endpoint, check-in and check-out date, number of adults and children. For possible filters use **@Filters for search** endpoint. Use `page_number` to paginate and navigate through the result pages! <br/> Comparable to: ![Search locations](https://i.imgur.com/PCuK81r.png)"
    adults_number: Number of adults
        checkin_date: Checkin date
        dest_id: Destination id, use `Search locations` to find a place, field `dest_id` and `dest_type`
        checkout_date: Checkout date
        room_number: Number of rooms
        include_adjacency: Include nearby places. If there are few hotels in the selected location, nearby locations will be added. You should pay attention to the `primary_count` parameter - it is the number of hotels from the beginning of the array that matches the strict filter.
        children_number: Number of children
        page_number: Page number
        children_ages: The age of every child. If children will be staying in the room, please indicate their ages separated by commas. 0-less than a year
        categories_filter_ids: Search filter IDs. For possible filters use `@Filters for search` ex: `price::USD-140-190`, or `price::USD-150-min` or `price::USD-150-max`
        
    """
    url = f"https://booking-com.p.rapidapi.com/v2/hotels/search"
    querystring = {'order_by': order_by, 'adults_number': adults_number, 'checkin_date': checkin_date, 'filter_by_currency': filter_by_currency, 'dest_id': dest_id, 'locale': locale, 'checkout_date': checkout_date, 'units': units, 'room_number': room_number, 'dest_type': dest_type, }
    if include_adjacency:
        querystring['include_adjacency'] = include_adjacency
    if children_number:
        querystring['children_number'] = children_number
    if page_number:
        querystring['page_number'] = page_number
    if children_ages:
        querystring['children_ages'] = children_ages
    if categories_filter_ids:
        querystring['categories_filter_ids'] = categories_filter_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def meta_properties_description(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get meta properties description"
    
    """
    url = f"https://booking-com.p.rapidapi.com/v2/hotels/meta-properties"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filters_for_search(checkin_date: str, locale: str, room_number: int, dest_id: int, checkout_date: str, adults_number: int, filter_by_currency: str, dest_type: str, order_by: str, units: str, include_adjacency: bool=None, categories_filter_ids: str='class::2,class::4,free_cancellation::1', page_number: int=0, children_number: int=2, children_ages: str='5,0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of available filters for hotel search"
    checkin_date: Checkin date
        room_number: Number of rooms
        dest_id: Destination id, use `Search locations` to find a place, field `dest_id` and `dest_type`
        checkout_date: Checkout date
        adults_number: Number of adults
        include_adjacency: Include nearby places. If there are few hotels in the selected location, nearby locations will be added. You should pay attention to the `primary_count` parameter - it is the number of hotels from the beginning of the array that matches the strict filter.
        categories_filter_ids: Search filter IDs. For possible filters use `@Filters for search` ex: `price::USD-140-190`, or `price::USD-150-min` or `price::USD-150-max`
        page_number: Page number
        children_number: Number of children
        children_ages: The age of every child. If children will be staying in the room, please indicate their ages separated by commas. 0-less than a year
        
    """
    url = f"https://booking-com.p.rapidapi.com/v2/hotels/search-filters"
    querystring = {'checkin_date': checkin_date, 'locale': locale, 'room_number': room_number, 'dest_id': dest_id, 'checkout_date': checkout_date, 'adults_number': adults_number, 'filter_by_currency': filter_by_currency, 'dest_type': dest_type, 'order_by': order_by, 'units': units, }
    if include_adjacency:
        querystring['include_adjacency'] = include_adjacency
    if categories_filter_ids:
        querystring['categories_filter_ids'] = categories_filter_ids
    if page_number:
        querystring['page_number'] = page_number
    if children_number:
        querystring['children_number'] = children_number
    if children_ages:
        querystring['children_ages'] = children_ages
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def description_of_the_hotel(locale: str, hotel_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get description of the hotel by hotel_id"
    hotel_id: Hotel id
        
    """
    url = f"https://booking-com.p.rapidapi.com/v2/hotels/description"
    querystring = {'locale': locale, 'hotel_id': hotel_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

