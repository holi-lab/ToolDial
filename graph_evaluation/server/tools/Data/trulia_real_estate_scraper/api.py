import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_for_rent(search_token: str, page: int, rental_type: str=None, sort: str=None, max_price: str=None, min_price: str=None, beds: str=None, dogs_allowed: bool=None, cats_allowed: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To download data from SALE, you must first generate a Token with Get search token and set search_type=ForRent. Then you need to use this generated token by setting the search_token value in this endpoint. Or just use one of our developer libraries. Everything is already implemented there."
    
    """
    url = f"https://trulia-real-estate-scraper.p.rapidapi.com/search/for_rent"
    querystring = {'search_token': search_token, 'page': page, }
    if rental_type:
        querystring['rental_type'] = rental_type
    if sort:
        querystring['sort'] = sort
    if max_price:
        querystring['max_price'] = max_price
    if min_price:
        querystring['min_price'] = min_price
    if beds:
        querystring['beds'] = beds
    if dogs_allowed:
        querystring['dogs_allowed'] = dogs_allowed
    if cats_allowed:
        querystring['cats_allowed'] = cats_allowed
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trulia-real-estate-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_sold(page: int, search_token: str, beds: str=None, sold_date: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To download data from SALE, you must first generate a Token with Get search token and set search_type=Sold. Then you need to use this generated token by setting the search_token value in this endpoint. Or just use one of our developer libraries. Everything is already implemented there."
    
    """
    url = f"https://trulia-real-estate-scraper.p.rapidapi.com/search/for_sold"
    querystring = {'page': page, 'search_token': search_token, }
    if beds:
        querystring['beds'] = beds
    if sold_date:
        querystring['sold_date'] = sold_date
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trulia-real-estate-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_for_sale(page: int, search_token: str, for_sale_by_owner: bool=None, min_price: str=None, for_sale_by_agent: bool=None, house_type: str=None, max_price: str=None, sort: str=None, beds: str=None, new_construction: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To download data from SALE, you must first generate a Token with Get search token and set search_type=ForSale. Then you need to use this generated token by setting the search_token value in this endpoint. Or just use one of our developer libraries. Everything is already implemented there."
    search_token: Call **Get search token** endpoint to generate the token and put it here.
        
    """
    url = f"https://trulia-real-estate-scraper.p.rapidapi.com/search/for_sale"
    querystring = {'page': page, 'search_token': search_token, }
    if for_sale_by_owner:
        querystring['for_sale_by_owner'] = for_sale_by_owner
    if min_price:
        querystring['min_price'] = min_price
    if for_sale_by_agent:
        querystring['for_sale_by_agent'] = for_sale_by_agent
    if house_type:
        querystring['house_type'] = house_type
    if max_price:
        querystring['max_price'] = max_price
    if sort:
        querystring['sort'] = sort
    if beds:
        querystring['beds'] = beds
    if new_construction:
        querystring['new_construction'] = new_construction
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trulia-real-estate-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_server_time(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns server time"
    
    """
    url = f"https://trulia-real-estate-scraper.p.rapidapi.com/GetServerTime"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trulia-real-estate-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_home_details(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns full details of home. Call **Get listing by url** or get items from *Search*. In response you'll get **url** of home. Take this url and pass it here into query. You can also go to https://www.trulia.com/AZ/Scottsdale/ and take urls e.g. https://www.trulia.com/p/az/fountain-hills/14834-e-valley-vista-dr-fountain-hills-az-85268--2113652369"
    
    """
    url = f"https://trulia-real-estate-scraper.p.rapidapi.com/homes/details_by_url"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trulia-real-estate-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_listing_by_url(page: int, url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns homes from the listing. Just go to https://www.trulia.com/ select the listing you are interested in e.g. https://www.trulia.com/AZ/Scottsdale/ and pass that url into query."
    
    """
    url = f"https://trulia-real-estate-scraper.p.rapidapi.com/homes/listing_by_url"
    querystring = {'page': page, 'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trulia-real-estate-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_search_token(search_type: str, place: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get search token and use it in **search/for_sale**, **search/for_rent**, **search/sold** endpoints."
    place: Type city (or phrase) you are interested in, example:  **California**, **Cali**, **Boston** etc.
        
    """
    url = f"https://trulia-real-estate-scraper.p.rapidapi.com/search/token"
    querystring = {'search_type': search_type, 'place': place, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "trulia-real-estate-scraper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

