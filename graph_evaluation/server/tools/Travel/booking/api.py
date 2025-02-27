import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def reviews_get_scores(hotel_ids: int, languagecode: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get reviewing scores"
    hotel_ids: The value of hotel_id field from properties/list API
        languagecode: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/reviews/get-scores"
    querystring = {'hotel_ids': hotel_ids, }
    if languagecode:
        querystring['languagecode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def review_filters_list(hotel_id: int, languagecode: str='en-us', filter_language: str='en,nl', filter_customer_type: str='couple,family_with_children', user_sort: str='sort_most_relevant', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List supported options metadata for filtering reviews"
    hotel_id: The value of hotel_id field from properties/list API
        languagecode: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        filter_language: One of the followings (separated by comma for multiple values): en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        filter_customer_type: One of the followings (separated by comma for multiple values) : couple|family_with_children|review_category_group_of_friends|solo_traveller
        user_sort: One of the followings : sort_most_relevant|sort_recent_desc|sort_recent_asc|sort_score_desc|sort_score_asc
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/review-filters/list"
    querystring = {'hotel_id': hotel_id, }
    if languagecode:
        querystring['languagecode'] = languagecode
    if filter_language:
        querystring['filter_language'] = filter_language
    if filter_customer_type:
        querystring['filter_customer_type'] = filter_customer_type
    if user_sort:
        querystring['user_sort'] = user_sort
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def filters_list(room_qty: int, departure_date: str, dest_ids: int, guest_qty: str, arrival_date: str, search_type: str, languagecode: str='en-us', categories_filter: str='price::9-40,free_cancellation::1,class::1,class::0,class::2', children_qty: str='2', children_age: str='5,7', longitude: int=None, price_filter_currencycode: str='USD', latitude: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A list of filter options to pass in categories_filter field. Pass params and values are as same as .../properties/list API"
    room_qty: The number of rooms
        departure_date: The check-out date 
        dest_ids: Value of dest_id or city_ufi field from locations/auto-complete API (Don't pass this if you use latlong as search_type)
        guest_qty: The number of adults
        arrival_date: The check-in date at hotel
        search_type: Value of dest_type returned by locations/auto-complete API
        languagecode: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        categories_filter: The id returned by .../filters/list API
        children_qty: The number of children
        children_age: The year old of each child that separated by comma
        longitude: 106.686102 - Don't pass this param if you DON'T use latlong as search_type
        price_filter_currencycode: The base currency to calculate exchange rate
        latitude: 10.838039 - Don't pass this param if you DON'T use latlong as search_type
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/filters/list"
    querystring = {'room_qty': room_qty, 'departure_date': departure_date, 'dest_ids': dest_ids, 'guest_qty': guest_qty, 'arrival_date': arrival_date, 'search_type': search_type, }
    if languagecode:
        querystring['languagecode'] = languagecode
    if categories_filter:
        querystring['categories_filter'] = categories_filter
    if children_qty:
        querystring['children_qty'] = children_qty
    if children_age:
        querystring['children_age'] = children_age
    if longitude:
        querystring['longitude'] = longitude
    if price_filter_currencycode:
        querystring['price_filter_currencycode'] = price_filter_currencycode
    if latitude:
        querystring['latitude'] = latitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_list(hotel_ids: int, languagecode: str='en-us', user_sort: str='sort_most_relevant', rows: int=25, offset: int=0, filter_language: str='en,nl', filter_customer_type: str='couple,family_with_children', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List reviews of stayed guests"
    hotel_ids: The value of hotel_id field from properties/list API
        languagecode: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        user_sort: One of the followings : sort_most_relevant|sort_recent_desc|sort_recent_asc|sort_score_desc|sort_score_asc
        rows: The number of items per page
        offset: The number of items to ignore as offset for paging 
        filter_language: One of the followings (separated by comma for multiple values): en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        filter_customer_type: One of the followings (separated by comma for multiple values) : couple|family_with_children|review_category_group_of_friends|solo_traveller
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/reviews/list"
    querystring = {'hotel_ids': hotel_ids, }
    if languagecode:
        querystring['languagecode'] = languagecode
    if user_sort:
        querystring['user_sort'] = user_sort
    if rows:
        querystring['rows'] = rows
    if offset:
        querystring['offset'] = offset
    if filter_language:
        querystring['filter_language'] = filter_language
    if filter_customer_type:
        querystring['filter_customer_type'] = filter_customer_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_hotel_photos(hotel_ids: int, languagecode: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get photos of hotel"
    hotel_ids: Get value from properties/list
        languagecode: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/properties/get-hotel-photos"
    querystring = {'hotel_ids': hotel_ids, }
    if languagecode:
        querystring['languagecode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_detail(rec_room_qty: int, search_id: str, departure_date: str, hotel_id: str, rec_guest_qty: int, arrival_date: str, block_ids: str=None, dest_ids: int=-3727579, units: str='imperial', currency_code: str='USD', rec_children_qty: int=None, recommend_for: int=3, rec_children_age: str=None, languagecode: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brief information of a property"
    rec_room_qty: The number of rooms
        search_id: The value returned in properties/list API having response contains the selected hotel
        departure_date: The check-out date 
        hotel_id: The value of hotel_id field from properties/list API
        rec_guest_qty: The number of adults separated by comma, and how you arrange them in rooms
        arrival_date: The check-in date at hotel
        block_ids: The value of 'block_ids' field returned in .../properties/list endpoint
        dest_ids: The value of 'dest_id' field returned in .../locations/auto-complete endpoint
        units: One of the following : metric|imperial
        currency_code: The currency code
        rec_children_qty: The number of children separated by comma, and how you arrange them in rooms
        recommend_for: The recommended number of persons per room
        rec_children_age: The year old of each child that separated by comma. Ex : 5,7
        languagecode: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/properties/detail"
    querystring = {'rec_room_qty': rec_room_qty, 'search_id': search_id, 'departure_date': departure_date, 'hotel_id': hotel_id, 'rec_guest_qty': rec_guest_qty, 'arrival_date': arrival_date, }
    if block_ids:
        querystring['block_ids'] = block_ids
    if dest_ids:
        querystring['dest_ids'] = dest_ids
    if units:
        querystring['units'] = units
    if currency_code:
        querystring['currency_code'] = currency_code
    if rec_children_qty:
        querystring['rec_children_qty'] = rec_children_qty
    if recommend_for:
        querystring['recommend_for'] = recommend_for
    if rec_children_age:
        querystring['rec_children_age'] = rec_children_age
    if languagecode:
        querystring['languagecode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_v2_get_rooms(rec_room_qty: str, arrival_date: str, rec_guest_qty: str, hotel_id: str, departure_date: str, units: str='imperial', currency_code: str='USD', block_ids: str=None, languagecode: str='en-us', rec_children_qty: str=None, rec_children_age: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full details of rooms in the hotel"
    arrival_date: The check-in date at hotel
        rec_guest_qty: The number of adults separated by comma, and how you arrange them in rooms
        hotel_id: The value of hotel_id field from properties/list API
        departure_date: The check-out date 
        units: One of the following : metric|imperial
        currency_code: The currency code
        block_ids: The value of 'block_ids' field returned in .../properties/list endpoint
        languagecode: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        rec_children_qty: The number of children separated by comma, and how you arrange them in rooms
        rec_children_age: The year old of each child that separated by comma. Ex : 5,12
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/properties/v2/get-rooms"
    querystring = {'rec_room_qty': rec_room_qty, 'arrival_date': arrival_date, 'rec_guest_qty': rec_guest_qty, 'hotel_id': hotel_id, 'departure_date': departure_date, }
    if units:
        querystring['units'] = units
    if currency_code:
        querystring['currency_code'] = currency_code
    if block_ids:
        querystring['block_ids'] = block_ids
    if languagecode:
        querystring['languagecode'] = languagecode
    if rec_children_qty:
        querystring['rec_children_qty'] = rec_children_qty
    if rec_children_age:
        querystring['rec_children_age'] = rec_children_age
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_policies(hotel_ids: int, languagecode: str='en-us', currency_code: str='USD', departure_date: str='2019-03-15', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get policies of property by id"
    hotel_ids: The value of hotel_id field from properties/list API
        languagecode: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        currency_code: The currency code
        departure_date: The check-out date at hotel
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/properties/get-policies"
    querystring = {'hotel_ids': hotel_ids, }
    if languagecode:
        querystring['languagecode'] = languagecode
    if currency_code:
        querystring['currency_code'] = currency_code
    if departure_date:
        querystring['departure_date'] = departure_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_featured_reviews(hotel_id: str, languagecode: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get featured reviews of stayed guests"
    hotel_id: The value of hotel_id from properties/list API
        languagecode: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/properties/get-featured-reviews"
    querystring = {'hotel_id': hotel_id, }
    if languagecode:
        querystring['languagecode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_description(hotel_ids: int, check_out: str='2019-03-15', languagecode: str='en-us', check_in: str='2019-03-13', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get description of property by id"
    hotel_ids: The value of hotel_id from properties/list API
        check_out: The check-out date
        languagecode: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        check_in: The check-in date at hotel
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/properties/get-description"
    querystring = {'hotel_ids': hotel_ids, }
    if check_out:
        querystring['check_out'] = check_out
    if languagecode:
        querystring['languagecode'] = languagecode
    if check_in:
        querystring['check_in'] = check_in
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list_by_map(arrival_date: str, room_qty: str, departure_date: str, guest_qty: str, bbox: str, travel_purpose: str='leisure', categories_filter: str='class::1,class::2,class::3', children_qty: int=2, search_id: str='none', children_age: str='11,5', price_filter_currencycode: str='USD', languagecode: str='en-us', order_by: str='popularity', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties by coordinate of bounding box"
    arrival_date: The check-in date at hotel, the format is yyyy-MM-dd. Ex : 2022-08-14
        room_qty: The number of rooms
        departure_date: The check-out date, the format is yyyy-MM-dd. Ex : 2022-08-15
        guest_qty: The number of adults
        bbox: Coordinate of bounding box - latitude (bottom left),latitude (top right),longitude (bottom left),longitude (top right)
        travel_purpose: One of the followings : leisure|business
        categories_filter: The id returned by .../filters/list API, values separated by comma
        children_qty: The number of children
        search_id: Pass \\\"none\\\" or value returned right in this API
        children_age: The year old of each child that separated by comma
        price_filter_currencycode: The base currency to calculate exchange rate
        languagecode: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        order_by: One of the followings : popularity|distance|class&#95;descending|class&#95;ascending|deals|review&#95;score|price
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/properties/list-by-map"
    querystring = {'arrival_date': arrival_date, 'room_qty': room_qty, 'departure_date': departure_date, 'guest_qty': guest_qty, 'bbox': bbox, }
    if travel_purpose:
        querystring['travel_purpose'] = travel_purpose
    if categories_filter:
        querystring['categories_filter'] = categories_filter
    if children_qty:
        querystring['children_qty'] = children_qty
    if search_id:
        querystring['search_id'] = search_id
    if children_age:
        querystring['children_age'] = children_age
    if price_filter_currencycode:
        querystring['price_filter_currencycode'] = price_filter_currencycode
    if languagecode:
        querystring['languagecode'] = languagecode
    if order_by:
        querystring['order_by'] = order_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_list(offset: int, departure_date: str, room_qty: int, guest_qty: int, arrival_date: str, search_type: str, dest_ids: int, price_filter_currencycode: str='USD', travel_purpose: str='leisure', latitude: str=None, longitude: str=None, categories_filter: str=None, order_by: str='popularity', children_age: str='5,7', children_qty: int=2, search_id: str='none', languagecode: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties having type of resorts, hotels, motels, hostels, etc as on official site"
    offset: The number of items to ignore as offset for paging (fixed 30 items returned per page). You are interested in the optional parameter 'search_id'
        departure_date: The check-out date, the format is yyyy-MM-dd. Ex : 2022-08-15
        room_qty: The number of rooms
        guest_qty: The number of adults
        arrival_date: The check-in date at hotel, the format is yyyy-MM-dd. Ex : 2022-08-14
        search_type: Value of dest_type returned by locations/auto-complete API
        dest_ids: Value of dest&#95;id or city&#95;ufi field from locations/auto-complete API (Don't pass this if you use latlong as search&#95;type)
        price_filter_currencycode: The base currency to calculate exchange rate
        travel_purpose: One of the followings : leisure|business
        latitude: 10.838039 - Don't pass this param if you DON'T use latlong as search_type
        longitude: 106.686102 - Don't pass this param if you DON'T use latlong as search_type
        categories_filter: The id returned by .../filters/list API, separated by comma for multiple options. Ex : price::9-40,free_cancellation::1,class::1,class::0,class::2
        order_by: One of the followings : popularity|distance|class&#95;descending|class&#95;ascending|deals|review&#95;score|price
        children_age: The year old of each child that separated by comma
        children_qty: The number of children
        search_id: Pass 'none' or value returned right in this API. You need to pass this parameter to get 'offset' working
        languagecode: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/properties/list"
    querystring = {'offset': offset, 'departure_date': departure_date, 'room_qty': room_qty, 'guest_qty': guest_qty, 'arrival_date': arrival_date, 'search_type': search_type, 'dest_ids': dest_ids, }
    if price_filter_currencycode:
        querystring['price_filter_currencycode'] = price_filter_currencycode
    if travel_purpose:
        querystring['travel_purpose'] = travel_purpose
    if latitude:
        querystring['latitude'] = latitude
    if longitude:
        querystring['longitude'] = longitude
    if categories_filter:
        querystring['categories_filter'] = categories_filter
    if order_by:
        querystring['order_by'] = order_by
    if children_age:
        querystring['children_age'] = children_age
    if children_qty:
        querystring['children_qty'] = children_qty
    if search_id:
        querystring['search_id'] = search_id
    if languagecode:
        querystring['languagecode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_static_map(width: int, longitude: int, zoom: int, height: int, latitude: int, currency_code: str='USD', languagecode: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get static map at specific GEO location"
    width: The width of image in pixel
        longitude: The latitude co-ordinate
        zoom: Zoom in/out of map
        height: The height of image in pixel
        latitude: The latitude co-ordinate
        currency_code: The currency code
        languagecode: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja\",         \"ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/properties/get-static-map"
    querystring = {'width': width, 'longitude': longitude, 'zoom': zoom, 'height': height, 'latitude': latitude, }
    if currency_code:
        querystring['currency_code'] = currency_code
    if languagecode:
        querystring['languagecode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_rooms_deprecated(search_id: str, arrival_date: str, rec_guest_qty: str, hotel_id: str, departure_date: str, travel_purpose: str='leisure', rec_children_age: str='5,7', recommend_for: int=3, languagecode: str='en-us', rec_children_qty: str='1,1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get full details of rooms in the hotel"
    search_id: The value returned in properties/list API having response contains the selected hotel
        arrival_date: The check-in date at hotel
        rec_guest_qty: The number of adults separated by comma, and how you arrange them in rooms
        hotel_id: The value of hotel_id field from properties/list API
        departure_date: The check-out date 
        travel_purpose: One of the followings : leisure|business
        rec_children_age: The year old of each child that separated by comma
        recommend_for: The recommended number of persons per room
        languagecode: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        rec_children_qty: The number of children separated by comma, and how you arrange them in rooms
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/properties/get-rooms"
    querystring = {'search_id': search_id, 'arrival_date': arrival_date, 'rec_guest_qty': rec_guest_qty, 'hotel_id': hotel_id, 'departure_date': departure_date, }
    if travel_purpose:
        querystring['travel_purpose'] = travel_purpose
    if rec_children_age:
        querystring['rec_children_age'] = rec_children_age
    if recommend_for:
        querystring['recommend_for'] = recommend_for
    if languagecode:
        querystring['languagecode'] = languagecode
    if rec_children_qty:
        querystring['rec_children_qty'] = rec_children_qty
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_facilities(hotel_ids: int, languagecode: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get facilities of property by id"
    hotel_ids: The value of hotel_id field from properties/list API
        languagecode: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/properties/get-facilities"
    querystring = {'hotel_ids': hotel_ids, }
    if languagecode:
        querystring['languagecode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currency_get_exchange_rates(base_currency: str, languagecode: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get currency exchange rates between different countries"
    base_currency: One of the followings : en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja|ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        languagecode: The language code of specific country
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/currency/get-exchange-rates"
    querystring = {'base_currency': base_currency, }
    if languagecode:
        querystring['languagecode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_auto_complete(text: str, languagecode: str='en-us', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List suggested locations by countries, cities, districts, places name"
    text: Name of cities, districts, places
        languagecode: One of the followings en|en-us|ar|bg|ca|cs|da|de|el|es|es-ar|et|fi|fr|he|hr|hu|id|is|it|ja",         "ko|lt|lv|ms|nl|no|pl|pt|pt-br|ro|ru|sk|sl|sr|sv|th|tl|tr|uk|vi|zh|zh-tw
        
    """
    url = f"https://apidojo-booking-v1.p.rapidapi.com/locations/auto-complete"
    querystring = {'text': text, }
    if languagecode:
        querystring['languagecode'] = languagecode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "apidojo-booking-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

