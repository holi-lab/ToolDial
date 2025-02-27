import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def properties_list(area: str, listing_id: str=None, radius: int=None, listing_status: str=None, include_retirement_homes: str=None, minimum_beds: int=None, include_shared_ownership: str=None, maximum_beds: int=None, maximum_price: int=None, include_sold: int=None, order_by: str='age', include_rented: int=None, identifier: str='oxford', category: str='residential', property_type: str=None, ordering: str='descending', new_homes: str=None, floor_area_max: int=None, floor_area_min: int=None, step_back_used: int=None, bills_included: str=None, include_featured_properties: int=None, created_since: str=None, pets_allowed: str=None, include_shared_accommodation: str=None, page_number: int=1, floor_area_units: str=None, page_size: int=40, furnished: str=None, minimum_price: int=None, keywords: str=None, price_frequency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List properties for sale or rent with options and filters"
    area: The value of suggestions/value json object returned in .../auto-complete endpoint with listings as search&#95;type. You must use EXACTLY the value  returned by the endpoint. 
*'listing&#95;id'  OR 'area' parameter must be provided to get this endpoint working.
        listing_id: The value of listing&#95;id field returned right in this endpoint. 
*'listing&#95;id'  OR 'area' parameter must be provided to get this endpoint working.
        radius: The radius (miles) to look for properties (1 - 40)
        listing_status: One of the following sale | rent
        include_retirement_homes: One of the following yes | no
        minimum_beds: Min number of bed rooms (1 - 10)
        include_shared_ownership: One of the following yes | no
        maximum_beds: Max number of bed rooms (1 - 10)
        maximum_price: Maximum sale or rent price
        include_sold: One of the following 1 | 0
        order_by: One of the following age|price|price&#95;change|view&#95;count
        include_rented: One of the following 1 | 0
        identifier: The value of suggestions/identifier json object returned in .../auto-complete endpoint with listings as search_type. You must use EXACTLY the value returned by the endpoint.
        category: One of the following residential|commercial
        property_type: One of the following and separated by comma for multiple values : flats|farms&#95;land|terraced|semi&#95;detached|detached|bungalow|park&#95;home|offices|retail|industrial|hospitality|land
        ordering: One of the following ascending|descending
        new_homes: One of the following yes | no
        floor_area_max: Maximum floor area, only use with commercial category.
        floor_area_min: Minimum floor area, only use with commercial category.
        step_back_used: One of the following 1 | 0
        bills_included: One of the following yes | no
        include_featured_properties: One of the following 1 | 0
        created_since: The date time from which properties added. The format must be yyyy-MM-dd HH:mm:ss, Ex : 2020-09-16 15:00:00
        pets_allowed: One of the following yes | no
        include_shared_accommodation: One of the following yes | no
        page_number: The page index for paging purpose
        floor_area_units: One of the following sq&#95;feet|sq&#95;metres
        page_size: The number of items per response (max 40)
        furnished: One of the following furnished|part&#95;furnished|unfurnished
        minimum_price: Minimum sale or rent price
        keywords: Any word or term, ex : garden,wooden floors
        price_frequency: One of the following per&#95;month|per&#95;year
        
    """
    url = f"https://zoopla.p.rapidapi.com/properties/list"
    querystring = {'area': area, }
    if listing_id:
        querystring['listing_id'] = listing_id
    if radius:
        querystring['radius'] = radius
    if listing_status:
        querystring['listing_status'] = listing_status
    if include_retirement_homes:
        querystring['include_retirement_homes'] = include_retirement_homes
    if minimum_beds:
        querystring['minimum_beds'] = minimum_beds
    if include_shared_ownership:
        querystring['include_shared_ownership'] = include_shared_ownership
    if maximum_beds:
        querystring['maximum_beds'] = maximum_beds
    if maximum_price:
        querystring['maximum_price'] = maximum_price
    if include_sold:
        querystring['include_sold'] = include_sold
    if order_by:
        querystring['order_by'] = order_by
    if include_rented:
        querystring['include_rented'] = include_rented
    if identifier:
        querystring['identifier'] = identifier
    if category:
        querystring['category'] = category
    if property_type:
        querystring['property_type'] = property_type
    if ordering:
        querystring['ordering'] = ordering
    if new_homes:
        querystring['new_homes'] = new_homes
    if floor_area_max:
        querystring['floor_area_max'] = floor_area_max
    if floor_area_min:
        querystring['floor_area_min'] = floor_area_min
    if step_back_used:
        querystring['step_back_used'] = step_back_used
    if bills_included:
        querystring['bills_included'] = bills_included
    if include_featured_properties:
        querystring['include_featured_properties'] = include_featured_properties
    if created_since:
        querystring['created_since'] = created_since
    if pets_allowed:
        querystring['pets_allowed'] = pets_allowed
    if include_shared_accommodation:
        querystring['include_shared_accommodation'] = include_shared_accommodation
    if page_number:
        querystring['page_number'] = page_number
    if floor_area_units:
        querystring['floor_area_units'] = floor_area_units
    if page_size:
        querystring['page_size'] = page_size
    if furnished:
        querystring['furnished'] = furnished
    if minimum_price:
        querystring['minimum_price'] = minimum_price
    if keywords:
        querystring['keywords'] = keywords
    if price_frequency:
        querystring['price_frequency'] = price_frequency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def house_prices_get_area_stats(property_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get area stats"
    property_id: The value of property_id field returned in .../house-prices/estimate endpoint.
        
    """
    url = f"https://zoopla.p.rapidapi.com/house-prices/get-area-stats"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def house_prices_get_historic_listings(property_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get historic listings"
    property_id: The value of property_id field returned in .../house-prices/estimate endpoint.
        
    """
    url = f"https://zoopla.p.rapidapi.com/house-prices/get-historic-listings"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def house_prices_get_points_of_interest(property_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get nearest points of interest"
    property_id: The value of property_id field returned in .../house-prices/estimate endpoint.
        
    """
    url = f"https://zoopla.p.rapidapi.com/house-prices/get-points-of-interest"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def house_prices_estimate(area: str, property_type: str=None, page_number: int=1, order_by: str='address', page_size: int=40, ordering: str='descending', identifier: str='west-sussex/crawley/greenwich-close', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returned list of estimated house prices"
    area: The value of suggestions/value json object returned in .../auto-complete endpoint with properties as search_type. You must use EXACTLY the value  returned by the endpoint.
        property_type: One of the following detached|flat|terraced|semi&#95;detached
        page_number: The page index for paging purpose
        order_by: One of the following price&#95;paid|last&#95;sold|address|estimated&#95;value
        page_size: The number of items per response (max 40)
        ordering: One of the following ascending|descending
        identifier: The value of suggestions/identifier json object returned in .../auto-complete endpoint with properties as search_type. You must use EXACTLY the value returned by the endpoint.
        
    """
    url = f"https://zoopla.p.rapidapi.com/house-prices/estimate"
    querystring = {'area': area, }
    if property_type:
        querystring['property_type'] = property_type
    if page_number:
        querystring['page_number'] = page_number
    if order_by:
        querystring['order_by'] = order_by
    if page_size:
        querystring['page_size'] = page_size
    if ordering:
        querystring['ordering'] = ordering
    if identifier:
        querystring['identifier'] = identifier
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_broadband(listing_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get broadband information"
    listing_id: The value of listing_id field returned in .../properties/list endpoint
        
    """
    url = f"https://zoopla.p.rapidapi.com/properties/get-broadband"
    querystring = {'listing_id': listing_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_running_costs(listing_id: int, category: str='residential', section: str='to-rent', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get running costs"
    listing_id: The value of listing_id field returned in .../properties/list endpoint
        category: One of the following residential|commercial
        section: One of the following for-sale|to-rent
        
    """
    url = f"https://zoopla.p.rapidapi.com/properties/get-running-costs"
    querystring = {'listing_id': listing_id, }
    if category:
        querystring['category'] = category
    if section:
        querystring['section'] = section
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_area_stats(listing_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get area stats"
    listing_id: The value of listing_id field returned in .../properties/list endpoint
        
    """
    url = f"https://zoopla.p.rapidapi.com/properties/get-area-stats"
    querystring = {'listing_id': listing_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_get_nearby(listing_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get nearest points of interest"
    listing_id: The value of listing_id field returned in .../properties/list endpoint
        
    """
    url = f"https://zoopla.p.rapidapi.com/properties/get-nearby"
    querystring = {'listing_id': listing_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def auto_complete(search_term: str, search_type: str='listings', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get auto complete suggestion by term or phrase"
    search_type: One of the following properties|listings. Use  listings value to get suggestion for .../properties/list endpoint. Use properties value to get suggestion for .../house-prices/estimate endpoint.
        
    """
    url = f"https://zoopla.p.rapidapi.com/auto-complete"
    querystring = {'search_term': search_term, }
    if search_type:
        querystring['search_type'] = search_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def house_prices_get_sales_history(property_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sales history"
    property_id: The value of property_id field returned in .../house-prices/estimate endpoint.
        
    """
    url = f"https://zoopla.p.rapidapi.com/house-prices/get-sales-history"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def house_prices_get_running_costs(property_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get running costs"
    property_id: The value of property_id field returned in .../house-prices/estimate endpoint.
        
    """
    url = f"https://zoopla.p.rapidapi.com/house-prices/get-running-costs"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def house_prices_get_market_activity(area: str, identifier: str='west-sussex/crawley/greenwich-close', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market activity in an area"
    area: The value of suggestions/value json object returned in .../auto-complete endpoint with properties as search_type. You must use EXACTLY the value  returned by the endpoint.
        identifier: The value of suggestions/identifier json object returned in .../auto-complete endpoint with properties as search_type. You must use EXACTLY the value returned by the endpoint.
        
    """
    url = f"https://zoopla.p.rapidapi.com/house-prices/get-market-activity"
    querystring = {'area': area, }
    if identifier:
        querystring['identifier'] = identifier
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "zoopla.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

