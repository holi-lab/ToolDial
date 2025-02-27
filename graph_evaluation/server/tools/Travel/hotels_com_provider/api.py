import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hotel_details(domain: str, locale: str, hotel_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hotel details"
    hotel_id: Hotel ID
        
    """
    url = f"https://hotels-com-provider.p.rapidapi.com/v2/hotels/details"
    querystring = {'domain': domain, 'locale': locale, 'hotel_id': hotel_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_info(hotel_id: int, locale: str, domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hotel info"
    hotel_id: Hotel ID
        
    """
    url = f"https://hotels-com-provider.p.rapidapi.com/v2/hotels/info"
    querystring = {'hotel_id': hotel_id, 'locale': locale, 'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def regions_search(locale: str, query: str, domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search region, locations, city or hotel by name"
    query: Query. Live search
        
    """
    url = f"https://hotels-com-provider.p.rapidapi.com/v2/regions"
    querystring = {'locale': locale, 'query': query, 'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_slug_convert(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Hotel ID from Hotel Site slug, (https://hotels.com/ho219115) `ho219115` -> `1105156`"
    slug: Hotels Slug
        
    """
    url = f"https://hotels-com-provider.p.rapidapi.com/v2/meta/convert/slug-id"
    querystring = {'slug': slug, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def domains_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of Domains, available domains code, currencies and locales"
    
    """
    url = f"https://hotels-com-provider.p.rapidapi.com/v2/domains"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotels_search(domain: str, sort_order: str, locale: str, checkout_date: str, region_id: int, adults_number: int, checkin_date: str, accessibility: str=None, available_filter: str='SHOW_AVAILABLE_ONLY', meal_plan: str='FREE_BREAKFAST', guest_rating_min: int=8, price_min: int=10, page_number: int=1, children_ages: str='4,0,15', amenities: str='WIFI,PARKING', price_max: int=500, lodging_type: str='HOTEL,HOSTEL,APART_HOTEL', payment_type: str='PAY_LATER,FREE_CANCELLATION', star_rating_ids: str='3,4,5', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hotels by the filter. Set the `domain`, it is necessary for localization at the currency setting. (meta / Domains List). `accessibility` parameter, needed to get ONLY available hotels. Indicate the `region_id` -> use `Region Search`, check-in and check-out date, number of adults and children"
    checkout_date: Checkout Date
        region_id: Region Id -> use `Regions Search`
        adults_number: Number of adults
        checkin_date: Checkin Date
        accessibility: Accessibility. List: `SIGN_LANGUAGE_INTERPRETER`,`STAIR_FREE_PATH`,`SERVICE_ANIMAL`,`IN_ROOM_ACCESSIBLE`,`ROLL_IN_SHOWER`,`ACCESSIBLE_BATHROOM`,`ELEVATOR`,`ACCESSIBLE_PARKING`
        available_filter: Available filter. List: `SHOW_AVAILABLE_ONLY`
        meal_plan: Meal plan. List: `ALL_INCLUSIVE`,`FULL_BOARD`,`HALF_BOARD`,`FREE_BREAKFAST`
        guest_rating_min: Minimal guest rating. Ex. `9` -> +9
        price_min: Price min
        page_number: Number of page
        children_ages: The age of every child. Indicate their ages separated by commas. `4,0,15` -> three kids. `0` - less than a year
        amenities: Amenities. List: `SPA_ON_SITE`,`WIFI`,`HOT_TUB`,`FREE_AIRPORT_TRANSPORTATION`,`POOL`,`GYM`,`OCEAN_VIEW`,`WATER_PARK`,`BALCONY_OR_TERRACE`,`KITCHEN_KITCHENETTE`,`ELECTRIC_CAR`,`PARKING`,`CRIB`,`RESTAURANT_IN_HOTEL`,`PETS`,`WASHER_DRYER`,`CASINO`,`AIR_CONDITIONING`
        price_max: Price max
        lodging_type: Lodging type. List: `HOSTAL`,`APARTMENT`,`APART_HOTEL`,`CHALET`,`HOTEL`,`RYOKAN`,`BED_AND_BREAKFAST`,`HOSTEL`
        payment_type: Payment type. List: `GIFT_CARD`,`PAY_LATER`,`FREE_CANCELLATION`
        star_rating_ids: Hotel Star ratings
        
    """
    url = f"https://hotels-com-provider.p.rapidapi.com/v2/hotels/search"
    querystring = {'domain': domain, 'sort_order': sort_order, 'locale': locale, 'checkout_date': checkout_date, 'region_id': region_id, 'adults_number': adults_number, 'checkin_date': checkin_date, }
    if accessibility:
        querystring['accessibility'] = accessibility
    if available_filter:
        querystring['available_filter'] = available_filter
    if meal_plan:
        querystring['meal_plan'] = meal_plan
    if guest_rating_min:
        querystring['guest_rating_min'] = guest_rating_min
    if price_min:
        querystring['price_min'] = price_min
    if page_number:
        querystring['page_number'] = page_number
    if children_ages:
        querystring['children_ages'] = children_ages
    if amenities:
        querystring['amenities'] = amenities
    if price_max:
        querystring['price_max'] = price_max
    if lodging_type:
        querystring['lodging_type'] = lodging_type
    if payment_type:
        querystring['payment_type'] = payment_type
    if star_rating_ids:
        querystring['star_rating_ids'] = star_rating_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_rooms_offers(adults_number: int, checkout_date: str, domain: str, locale: str, hotel_id: int, checkin_date: str, children_ages: str='4,0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get available rooms at the hotel"
    adults_number: Number of adults
        checkout_date: Checkout date
        hotel_id: Hotel ID
        checkin_date: Checkin date
        children_ages: The age of every child. If children will be staying in the room, please indicate their ages separated by commas. 0-less than a year
        
    """
    url = f"https://hotels-com-provider.p.rapidapi.com/v2/hotels/offers"
    querystring = {'adults_number': adults_number, 'checkout_date': checkout_date, 'domain': domain, 'locale': locale, 'hotel_id': hotel_id, 'checkin_date': checkin_date, }
    if children_ages:
        querystring['children_ages'] = children_ages
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_scores(domain: str, hotel_id: int, locale: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hotel overall score"
    hotel_id: Hotel ID
        
    """
    url = f"https://hotels-com-provider.p.rapidapi.com/v2/hotels/reviews/scores"
    querystring = {'domain': domain, 'hotel_id': hotel_id, 'locale': locale, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_list(domain: str, locale: str, hotel_id: int, page_number: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hotel reviews list"
    hotel_id: Hotel ID
        page_number: Number of page
        
    """
    url = f"https://hotels-com-provider.p.rapidapi.com/v2/hotels/reviews/list"
    querystring = {'domain': domain, 'locale': locale, 'hotel_id': hotel_id, }
    if page_number:
        querystring['page_number'] = page_number
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_summary(locale: str, domain: str, hotel_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hotel reviews summary"
    hotel_id: Hotel ID
        
    """
    url = f"https://hotels-com-provider.p.rapidapi.com/v2/hotels/reviews/summary"
    querystring = {'locale': locale, 'domain': domain, 'hotel_id': hotel_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotel_summary(domain: str, locale: str, hotel_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get hotel summary"
    hotel_id: Hotel ID
        
    """
    url = f"https://hotels-com-provider.p.rapidapi.com/v2/hotels/summary"
    querystring = {'domain': domain, 'locale': locale, 'hotel_id': hotel_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotels-com-provider.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

