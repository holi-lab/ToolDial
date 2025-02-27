import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_stores_in_a_city(state: str, api_key: str, city: str, brand: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all stores in the city, [use endpoint /api/v1.1/cities/ to get all city names]                
		    Filter by city,state,brand_alias
		Example Payload
		 {
		    "api_key":"iopII2344_ADDDxxw1i1",
		    "city": "Los Angeles",
		    "state": "CA",
		    "brand": "taco-bell" (Optional)
		  }
		Example Response
		 [{
		    "brand_name": "Western Union",
		    "store_address": "600 West 7th St",
		    "phone_no": "(213) 896-0083"
		  },{
		    "brand_name": "Simple Mobile",
		    "store_address": "727 N Vine St",
		    "phone_no": "(323) 466-7300"
		  }]"
    
    """
    url = f"https://stores-and-brands-api.p.rapidapi.com/stores-in-city/"
    querystring = {'state': state, 'api_key': api_key, 'city': city, }
    if brand:
        querystring['brand'] = brand
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stores-and-brands-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def stores_near_me(radius: int, lat: str, long: str, api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find Stores Near Me by sending lat and long 
		    Get all stores in the city, [use endpoint /api/v1.1/cities/ to get all city names]                
		    Filter by city,state,brand_alias
		Example Payload
		 {
		    "api_key":"iopII2344_ADDDxxw1i1",
		    "lat": "12.34",
		    "long": "11.55",
		    "radius", 10 (optional in kms)
		  }
		Example Response
		 [{
		    "brand_name": "Western Union",
		    "store_address": "600 West 7th St",
		    "phone_no": "(213) 896-0083"
		  },{
		    "brand_name": "Simple Mobile",
		    "store_address": "727 N Vine St",
		    "phone_no": "(323) 466-7300"
		  }]"
    
    """
    url = f"https://stores-and-brands-api.p.rapidapi.com/stores-near-me/"
    querystring = {'radius': radius, 'lat': lat, 'long': long, 'api_key': api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stores-and-brands-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_stores_by_city(api_key: str, state: str, brand_alias: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all cities and store counts in a particular state
		    Use brand filter to get cities for the brand
		
		Example Payload
		 {
		    "api_key":"iopII2344_ADDDxxw1i1",
		    "state":"CA",                    
		    "brand_alias": "western-union" (optional)
		 }
		Example Response
		 [{
		    
		    "city": "Los Angeles",
		    "store_count": 630,
		    "state","CA"
		  },
		  {
		   
		    "city": "Newyork",
		    "store_count": 1200,
		    "state":"NY"
		  }]"
    
    """
    url = f"https://stores-and-brands-api.p.rapidapi.com/cities/"
    querystring = {'api_key': api_key, 'state': state, }
    if brand_alias:
        querystring['brand_alias'] = brand_alias
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stores-and-brands-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_brand_stores(api_key: str, brand_alias: str, radius: int=None, long: str=None, state: str=None, lat: str=None, city: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all brand stores, Use brand_alias 
		 from /get-all-brands/ endpoint to get brand stores
		 You can also filter by lat,lon,radius(in km),city,state
		Example Payload
		 {
		    "api_key":"iopII2344_ADDDxxw1i1",
		    "brand_alias": "western-union"
		    "lat": "34.0477542", (optional)
		    "long": "-118.2576388", (optional)
		    "radius": "10", (optional)
		    "city":"Los Angeles",
		    "state":"CA"
		 }
		Example Response
		 {
		    "brand_name": "Western Union",
		    "store_address": "600 West 7th St",
		    "phone_no": "(213) 896-0083",
		    "city": "Los Angeles",
		    "state": "CA",
		    "country": "US",
		    "code": 200
		  }"
    
    """
    url = f"https://stores-and-brands-api.p.rapidapi.com/brand-stores/"
    querystring = {'api_key': api_key, 'brand_alias': brand_alias, }
    if radius:
        querystring['radius'] = radius
    if long:
        querystring['long'] = long
    if state:
        querystring['state'] = state
    if lat:
        querystring['lat'] = lat
    if city:
        querystring['city'] = city
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stores-and-brands-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_brands(api_key: str, limit: int=100, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get all stores from FindOpenHours you can also filter by [categories](https://findopenhours.org/all-categories/).
		Example Payload
		 {
		  "api_key":"iopII2344_ADDDxxw1i1",
		  "page":0,
		  "limit":200
		  }
		Example Response
		 {
		    
		    "brand_name": "Western Union",
		    "brand_alias": "western-union",
		    "store_count": 39898,
		    "code": 200
		  }"
    
    """
    url = f"https://stores-and-brands-api.p.rapidapi.com/get-all-brands/"
    querystring = {'api_key': api_key, }
    if limit:
        querystring['limit'] = limit
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "stores-and-brands-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

