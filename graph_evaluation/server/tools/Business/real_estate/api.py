import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def properties_rent(city: str, state: str, page: str, beds: str=None, status: str=None, sort: str='newest', type: str=None, price_h: str='2000', baths: str=None, price_l: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "U.S properties for rent"
    page: Enter the starting page number, default (1)
        beds: Minimum number of beds
        status: Choose one: `foreclosure` |`hide-new-constuction`|`show-new-constuction`|`show-55-plus`
        sort: Choose one: `new_listing`|`newest`|`affordable`|`luxury`
        type: Choose from the following: `single-family,multi-family,mobile,farm,condo,townhome,land`
        price_h: Filter by the maximum price of the property
        baths: Minimum number of baths
        price_l: Filter by the minimum price of the property
        
    """
    url = f"https://real-estate12.p.rapidapi.com/listings/rent"
    querystring = {'city': city, 'state': state, 'page': page, }
    if beds:
        querystring['beds'] = beds
    if status:
        querystring['status'] = status
    if sort:
        querystring['sort'] = sort
    if type:
        querystring['type'] = type
    if price_h:
        querystring['price_h'] = price_h
    if baths:
        querystring['baths'] = baths
    if price_l:
        querystring['price_l'] = price_l
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-estate12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def property_details(property_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get detailed information on U.S properties"
    property_id: Enter property ID. (This information can be found on sales and rent endpoints results)
        
    """
    url = f"https://real-estate12.p.rapidapi.com/listings-details"
    querystring = {'property_id': property_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-estate12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def properties_sale(state: str, city: str, page: str, sort: str='relevant', price_l: str=None, status: str=None, price_h: str=None, beds: str=None, type: str='single-family,multi-family', baths: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "U.S properties for sale"
    page: Enter the starting page number, default (1)
        sort: Choose one: `relevant`|`newest`|`price_low`|`price_high`|`open_house_date`|`price_reduce`|`largest_sqft`|`lot_size`
        price_l: Filter by the minimum price of the property
        status: Choose one: `foreclosure` |`hide-new-constuction`|`show-new-constuction`|`show-55-plus`
        price_h: Filter by the maximum price of the property
        beds: Minimum number of beds
        type: Choose from the following: `single-family,multi-family,mobile,farm,condo,townhome,land`
        baths: Minimum number of baths
        
    """
    url = f"https://real-estate12.p.rapidapi.com/listings/sale"
    querystring = {'state': state, 'city': city, 'page': page, }
    if sort:
        querystring['sort'] = sort
    if price_l:
        querystring['price_l'] = price_l
    if status:
        querystring['status'] = status
    if price_h:
        querystring['price_h'] = price_h
    if beds:
        querystring['beds'] = beds
    if type:
        querystring['type'] = type
    if baths:
        querystring['baths'] = baths
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "real-estate12.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

