import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product_details(tcin: str, store_id: str, zip: str='94611', state: str='CA', latitude: str='37.820', longitude: str='-122.200', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns detailed product information.
		Including:
		```
		product variants (with different dimension like size, color and etc.
		ratings and reviews
		item images/videos
		price
		promotion
		child products
		```"
    tcin: Target product id.
Value comes from product search API.
        store_id: The id of the store which product details data is being retrieved from. (Required, CANNOT be empty)
Value comes from nearby store API.

Example: 3330
        zip: User's zipcode. (optional)
        state: State code of user's location. (optional)
        latitude: User's latitude information. (optional)
        longitude: User's longitude information. (optional)
        
    """
    url = f"https://target-com-shopping-api.p.rapidapi.com/product_details"
    querystring = {'tcin': tcin, 'store_id': store_id, }
    if zip:
        querystring['zip'] = zip
    if state:
        querystring['state'] = state
    if latitude:
        querystring['latitude'] = latitude
    if longitude:
        querystring['longitude'] = longitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target-com-shopping-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_search(store_id: str, keyword: str, offset: str='0', count: str='25', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of products based on keyword."
    count: maximum 25
        
    """
    url = f"https://target-com-shopping-api.p.rapidapi.com/product_search"
    querystring = {'store_id': store_id, 'keyword': keyword, }
    if offset:
        querystring['offset'] = offset
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target-com-shopping-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nearby_stores(place: str, within: str='100', limit: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the list of stores near to specified ZIP code."
    within: Radius of searching distance in miles
        
    """
    url = f"https://target-com-shopping-api.p.rapidapi.com/nearby_stores"
    querystring = {'place': place, }
    if within:
        querystring['within'] = within
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target-com-shopping-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_fulfillment(tcin: str, accept: str='application/json', cache_control: str='no-cache', authority: str='redsky.target.com', accept_language: str='en-US,en;q=0.9', zip: str='94611', state: str='CA', longitude: str='-122.200', store_id: str='3330', latitude: str='37.820', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns product fulfillment information."
    tcin: Target product id.
Value comes from product search API.

        zip: The zipcode of user's search location.
(optional)
        state: State code where is user is located at. (optional)
        longitude: User's longitude Information (optional)
        store_id: The id of the Target store from which the fulfillment information is being retrieved.
Value comes from nearby stores api.
(optional)
        latitude: User's longitude Information (optional)
        
    """
    url = f"https://target-com-shopping-api.p.rapidapi.com/product_fulfillment"
    querystring = {'tcin': tcin, }
    if accept:
        querystring['accept'] = accept
    if cache_control:
        querystring['cache-control'] = cache_control
    if authority:
        querystring['authority'] = authority
    if accept_language:
        querystring['accept-language'] = accept_language
    if zip:
        querystring['zip'] = zip
    if state:
        querystring['state'] = state
    if longitude:
        querystring['longitude'] = longitude
    if store_id:
        querystring['store_id'] = store_id
    if latitude:
        querystring['latitude'] = latitude
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target-com-shopping-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_autocomplete(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Product autocompletion based on search keyword."
    
    """
    url = f"https://target-com-shopping-api.p.rapidapi.com/autocomplete"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target-com-shopping-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

