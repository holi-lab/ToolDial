import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def restaurants_get_info_deprecated(id_restaurant: int, locale: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available information of specific restaurant"
    id_restaurant: The value of data/id field returned in …/restaurants/list endpoint
        locale: The language code
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/restaurants/get-info"
    querystring = {'id_restaurant': id_restaurant, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def restaurants_auto_complete_deprecated(text: str, latitude: int=None, longitude: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestion by term or phrase"
    text: City, district,country name, etc any word that you are familiar with
        latitude: The end user's location, used for sorting purpose
        longitude: The end user's location, used for sorting purpose
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/restaurants/auto-complete"
    querystring = {'text': text, }
    if latitude:
        querystring['latitude'] = latitude
    if longitude:
        querystring['longitude'] = longitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_auto_complete_deprecated(text: str, latitude: int=None, longitude: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestion by term or phrase"
    text: City, district,country name, etc any word that you are familiar with
        latitude: The end user's location, used for sorting purpose
        longitude: The end user's location, used for sorting purpose
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/locations/auto-complete"
    querystring = {'text': text, }
    if latitude:
        querystring['latitude'] = latitude
    if longitude:
        querystring['longitude'] = longitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sale_type_menu_list_deprecated(id_restaurant: int, locale: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get menu of specific restaurant"
    id_restaurant: The value of data/id field returned in …/restaurants/v2/list endpoint
        locale: The language code
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/sale-type-menu/list"
    querystring = {'id_restaurant': id_restaurant, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_get_stat(id_restaurant: int, locale: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get review stat of specific restaurant"
    id_restaurant: The value of data/id field returned in …/restaurants/list endpoint
        locale: The language code
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/reviews/get-stat"
    querystring = {'id_restaurant': id_restaurant, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_list(id_restaurant: int, food_report: int=None, limit: int=15, page: int=1, sort: str='MEAL_DATE_DESC', locale: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List reviews of specific restaurant by diners"
    id_restaurant: The value of data/id field returned in …/restaurants/v2/list endpoint
        food_report: One of following is allowed 1|0
        limit: For paging purpose
        page: For paging purpose
        sort: One of following is allowed AVERAGE&#95;RATING&#95;DESC|AVERAGE&#95;RATING&#95;ASC|MEAL&#95;DATE&#95;DESC|MEAL&#95;DATE&#95;ASC
        locale: The language code
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/reviews/list"
    querystring = {'id_restaurant': id_restaurant, }
    if food_report:
        querystring['food_report'] = food_report
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_list_best(id_restaurant: int, locale: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all best reviews of specific restaurant by diners"
    id_restaurant: The value of data/id field returned in …/restaurants/v2/list endpoint
        locale: The language code
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/reviews/list-best"
    querystring = {'id_restaurant': id_restaurant, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def reviews_v2_list(restaurantid: int, withreview: str='WITH_REVIEW', offset: int=0, limit: int=40, language: str=None, orderby: str='MEAL_DATE', sortdirection: str='DESC', occasion: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List reviews of specific restaurant by diners"
    restaurantid: The value of data/id field returned in …/restaurants/v2/list endpoint
        withreview: One of the following : WITH_REVIEW|ALL
        offset: The offset of records to ignore for paging purpose.
        limit: For paging purpose
        language: One of the following : en|sv|it|fr|es|nl|de|pt|no|ru|da|lb
        orderby: One of following is allowed RATING|MEAL&#95;DATE
        sortdirection: One of following is allowed ASC|DESC
        occasion: One of the following : BUSINESS|FRIENDS|ROMANTIC|FAMILY|ALONE
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/reviews/v2/list"
    querystring = {'restaurantId': restaurantid, }
    if withreview:
        querystring['withReview'] = withreview
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if language:
        querystring['language'] = language
    if orderby:
        querystring['orderBy'] = orderby
    if sortdirection:
        querystring['sortDirection'] = sortdirection
    if occasion:
        querystring['occasion'] = occasion
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def food_report_list_deprecated(id_restaurant: int, locale: str='en_US', count: int=10, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available food reported by other diners"
    id_restaurant: The value of data/id field returned in …/restaurants/v2/list endpoint
        locale: The language code
        count: The number of items returned in each response
        page: For paging purpose

        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/food-report/list"
    querystring = {'id_restaurant': id_restaurant, }
    if locale:
        querystring['locale'] = locale
    if count:
        querystring['count'] = count
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sale_type_list_deprecated(id_restaurant: int, locale: str='en_US', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available sale type of specific restaurant"
    id_restaurant: The value of data/id field returned in …/restaurants/v2/list endpoint
        locale: The language code
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/sale-type/list"
    querystring = {'id_restaurant': id_restaurant, }
    if locale:
        querystring['locale'] = locale
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def restaurants_v2_list(queryplacevaluecityid: int, filterpricestart: str=None, queryplacevaluecoordinateslatitude: int=None, sort: str=None, querypromotiononly: bool=None, filterrestauranttagidlist: str=None, filtersaletypetagidlist: str=None, filterratestart: int=None, filtergroupedpromotion: str=None, pagenumber: int=1, queryplacevaluecoordinateslongitude: int=None, pagesize: int=10, filterpriceend: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List restaurants with options and filters"
    queryplacevaluecityid: The value of id_city field returned in …/locations/v2/list endpoint
        filterpricestart: Min price of meal
        queryplacevaluecoordinateslatitude: The latitude of GEO location to look for around restaurants. Ex : 45.4642035
        sort: One of following is allowed quality|promotion|price|popularity|avg&#95;rate|new&#95;restaurant
        querypromotiononly: false/true
        filterrestauranttagidlist: Look for suitable value of meta/tagCategories/tags/id returend right in this endpoint, separated by comma for multiple value. Ex : 387,513
        filtersaletypetagidlist: Look for suitable value of meta/filters/saleTypeTag/id returend right in this endpoint, separated by comma for multiple value. Ex : 1,3
        filterratestart: Min rate
        filtergroupedpromotion: One of following is allowed 20|25|30|40|50, separated by comma for multiple value. Ex : 20,25,30
        pagenumber: For paging purpose
        queryplacevaluecoordinateslongitude: The longitude of GEO location to look for around restaurants. Ex : 9.189982
        pagesize: For paging purpose
        filterpriceend: Max price of meal
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/restaurants/v2/list"
    querystring = {'queryPlaceValueCityId': queryplacevaluecityid, }
    if filterpricestart:
        querystring['filterPriceStart'] = filterpricestart
    if queryplacevaluecoordinateslatitude:
        querystring['queryPlaceValueCoordinatesLatitude'] = queryplacevaluecoordinateslatitude
    if sort:
        querystring['sort'] = sort
    if querypromotiononly:
        querystring['queryPromotionOnly'] = querypromotiononly
    if filterrestauranttagidlist:
        querystring['filterRestaurantTagIdList'] = filterrestauranttagidlist
    if filtersaletypetagidlist:
        querystring['filterSaleTypeTagIdList'] = filtersaletypetagidlist
    if filterratestart:
        querystring['filterRateStart'] = filterratestart
    if filtergroupedpromotion:
        querystring['filterGroupedPromotion'] = filtergroupedpromotion
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    if queryplacevaluecoordinateslongitude:
        querystring['queryPlaceValueCoordinatesLongitude'] = queryplacevaluecoordinateslongitude
    if pagesize:
        querystring['pageSize'] = pagesize
    if filterpriceend:
        querystring['filterPriceEnd'] = filterpriceend
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_list_deprecated(google_place_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List relating locations by Google place id"
    google_place_id: The value of data/geolocation/id/id fields returned in …/locations/auto-complete endpoint
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/locations/list"
    querystring = {'google_place_id': google_place_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_v2_list(google_place_id: str, geo_text: str='Roma, Metropolitan City of Rome, Italy', geo_ref: bool=None, geo_type: str='locality', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List relating locations by Google place id"
    google_place_id: The value of data/geolocation/id/id fields returned in …/locations/auto-complete endpoint
        geo_text: The value of 'text' fields returned in .../locations/v2/auto-complete endpoint
        geo_type: The value of 'type' fields returned in .../locations/v2/auto-complete
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/locations/v2/list"
    querystring = {'google_place_id': google_place_id, }
    if geo_text:
        querystring['geo_text'] = geo_text
    if geo_ref:
        querystring['geo_ref'] = geo_ref
    if geo_type:
        querystring['geo_type'] = geo_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def locations_v2_auto_complete(text: str, longitude: int=None, latitude: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestion for locations by term or phrase"
    text: City, district,country name, etc any word that you are familiar with
        longitude: The end user's location, used for sorting purpose
        latitude: The end user's location, used for sorting purpose
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/locations/v2/auto-complete"
    querystring = {'text': text, }
    if longitude:
        querystring['longitude'] = longitude
    if latitude:
        querystring['latitude'] = latitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def restaurants_v2_get_info(restaurantid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all available information of specific restaurant"
    restaurantid: The value of data/id field returned in …/restaurants/v2/list endpoint
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/restaurants/v2/get-info"
    querystring = {'restaurantId': restaurantid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def restaurants_list_deprecated(queryplacevaluecityid: int, sort: str=None, pagesize: int=10, filtersaletypetagidlist: str=None, queryplacevaluecoordinateslatitude: int=None, pagenumber: int=1, queryplacevaluecoordinateslongitude: int=None, filterrestauranttagidlist: str=None, filterpriceend: int=None, filterpricestart: str=None, filtergroupedpromotion: str=None, querypromotiononly: bool=None, filterratestart: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List restaurants with options and filters"
    queryplacevaluecityid: The value of id_city field returned in …/locations/list endpoint
        sort: One of following is allowed quality|promotion|price|popularity|avg&#95;rate|new&#95;restaurant
        pagesize: For paging purpose
        filtersaletypetagidlist: Look for suitable value of meta/filters/saleTypeTag/id returend right in this endpoint, separated by comma for multiple value. Ex : 1,3
        queryplacevaluecoordinateslatitude: The latitude of GEO location to look for around restaurants. Ex : 45.4642035
        pagenumber: For paging purpose
        queryplacevaluecoordinateslongitude: The longitude of GEO location to look for around restaurants. Ex : 9.189982
        filterrestauranttagidlist: Look for suitable value of meta/tagCategories/tags/id returend right in this endpoint, separated by comma for multiple value. Ex : 387,513
        filterpriceend: Max price of meal
        filterpricestart: Min price of meal
        filtergroupedpromotion: One of following is allowed 20|25|30|40|50, separated by comma for multiple value. Ex : 20,25,30
        querypromotiononly: false/true
        filterratestart: Min rate
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/restaurants/list"
    querystring = {'queryPlaceValueCityId': queryplacevaluecityid, }
    if sort:
        querystring['sort'] = sort
    if pagesize:
        querystring['pageSize'] = pagesize
    if filtersaletypetagidlist:
        querystring['filterSaleTypeTagIdList'] = filtersaletypetagidlist
    if queryplacevaluecoordinateslatitude:
        querystring['queryPlaceValueCoordinatesLatitude'] = queryplacevaluecoordinateslatitude
    if pagenumber:
        querystring['pageNumber'] = pagenumber
    if queryplacevaluecoordinateslongitude:
        querystring['queryPlaceValueCoordinatesLongitude'] = queryplacevaluecoordinateslongitude
    if filterrestauranttagidlist:
        querystring['filterRestaurantTagIdList'] = filterrestauranttagidlist
    if filterpriceend:
        querystring['filterPriceEnd'] = filterpriceend
    if filterpricestart:
        querystring['filterPriceStart'] = filterpricestart
    if filtergroupedpromotion:
        querystring['filterGroupedPromotion'] = filtergroupedpromotion
    if querypromotiononly:
        querystring['queryPromotionOnly'] = querypromotiononly
    if filterratestart:
        querystring['filterRateStart'] = filterratestart
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def restaurants_v2_auto_complete(text: str, longitude: int=None, latitude: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get suggestion for restaurants by term or phrase"
    text: City, district,country name, etc any word that you are familiar with
        longitude: The end user's location, used for sorting purpose
        latitude: The end user's location, used for sorting purpose
        
    """
    url = f"https://the-fork-the-spoon.p.rapidapi.com/restaurants/v2/auto-complete"
    querystring = {'text': text, }
    if longitude:
        querystring['longitude'] = longitude
    if latitude:
        querystring['latitude'] = latitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-fork-the-spoon.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

