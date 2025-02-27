import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_restaurant(restaurant_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Single Restaurant By ID"
    
    """
    url = f"https://documenu.p.rapidapi.com/restaurant/{restaurant_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "documenu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def menu_item(item_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Single Menu Item"
    item_id: ID of the menu item to get	
        
    """
    url = f"https://documenu.p.rapidapi.com/menuitem/{item_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "documenu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def menu_items_for_restaurant(restaurant_id: int, page: int=2, size: int=25, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns Menu Items By Restaurant ID"
    restaurant_id: Numeric ID of the restaurant to get.	
        page: Page Through Results	
        size: Data Size of Results	
        
    """
    url = f"https://documenu.p.rapidapi.com/restaurant/{restaurant_id}/menuitems"
    querystring = {}
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "documenu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_menu_items_geo(lat: int, lon: int, distance: int, size: int=30, cuisine: str='italian', search: str='pizza', page: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Function using location and search radius. Returns list of menu item sorted by closest"
    lat: Latitude of search area	
        lon: Longitude of the search area	
        distance: Search Area radius (in miles)	
        size: Number of Results To Return
        cuisine: Filter by cuisine
        search: Optional Terms to search for in menu item name and description	
        page: Page Through Results	
        
    """
    url = f"https://documenu.p.rapidapi.com/menuitems/search/geo"
    querystring = {'lat': lat, 'lon': lon, 'distance': distance, }
    if size:
        querystring['size'] = size
    if cuisine:
        querystring['cuisine'] = cuisine
    if search:
        querystring['search'] = search
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "documenu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_restaurants_fields(state: str=None, restaurant_name: str=None, exact: bool=None, zip_code: str=None, page: str=None, fullmenu: bool=None, restaurant_website: str=None, address: str=None, size: str=None, restaurant_phone: str='5854420444', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Restaurants By Fields"
    exact: Should search use exact match
        page: Page Through Results	
        fullmenu: include full menus
        size: Data Size of Results	
        
    """
    url = f"https://documenu.p.rapidapi.com/restaurants/search/fields"
    querystring = {}
    if state:
        querystring['state'] = state
    if restaurant_name:
        querystring['restaurant_name'] = restaurant_name
    if exact:
        querystring['exact'] = exact
    if zip_code:
        querystring['zip_code'] = zip_code
    if page:
        querystring['page'] = page
    if fullmenu:
        querystring['fullmenu'] = fullmenu
    if restaurant_website:
        querystring['restaurant_website'] = restaurant_website
    if address:
        querystring['address'] = address
    if size:
        querystring['size'] = size
    if restaurant_phone:
        querystring['restaurant_phone'] = restaurant_phone
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "documenu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def restaurants_by_state(state_code: str, top_cuisines: bool=None, cuisine: str='Italian', page: int=2, fullmenu: bool=None, size: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Function Returns Restaurants By State Code"
    top_cuisines: Show only Cuisines In Search Area	
        cuisine: Cuisine To Search For	
        page: Page Through Results	
        fullmenu: Include full menus
        size: Size of Results To Return
        
    """
    url = f"https://documenu.p.rapidapi.com/restaurants/state/{state_code}"
    querystring = {}
    if top_cuisines:
        querystring['top_cuisines'] = top_cuisines
    if cuisine:
        querystring['cuisine'] = cuisine
    if page:
        querystring['page'] = page
    if fullmenu:
        querystring['fullmenu'] = fullmenu
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "documenu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def restaurants_in_geo_bounding_box(top_left: str, bottom_right: str, top_cuisines: bool=None, cuisine: str='Italian', size: int=30, fullmenu: bool=None, page: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Function using Geobounding Box. Returns list of restaurants inside GeoBounding Box."
    top_left: Top Left coordinates of Bounding Box in lat,lon	
        bottom_right: Bottom Right coordinates of Bounding Box in lat,lon	
        top_cuisines: Show only Cuisines In Search Area	
        cuisine: Cuisine To Search For	
        size: Size of Results To Return
        fullmenu: Include full menus
        page: Page Through Results	
        
    """
    url = f"https://documenu.p.rapidapi.com/restaurants/geobbox"
    querystring = {'top_left': top_left, 'bottom_right': bottom_right, }
    if top_cuisines:
        querystring['top_cuisines'] = top_cuisines
    if cuisine:
        querystring['cuisine'] = cuisine
    if size:
        querystring['size'] = size
    if fullmenu:
        querystring['fullmenu'] = fullmenu
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "documenu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def restaurants_within_walking_driving_time(minutes: int, lon: int, mode: str, lat: int, top_cuisines: bool=None, cuisine: str='Italian', fullmenu: bool=None, page: int=2, size: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Function using the mode of transport and minutes to return a list of restaurants in that radius."
    minutes: Number of minutes Travel Time
        lon: Longitude of the search area	
        mode: Mode of Transport (walking or driving)	
        lat: Latitude of search area	
        top_cuisines: Show only Cuisines In Search Area	
        cuisine: cuisine
        fullmenu: Should Results Include full menus	
        page: Page Through Results	
        size: Number of Results To Return
        
    """
    url = f"https://documenu.p.rapidapi.com/restaurants/distance"
    querystring = {'minutes': minutes, 'lon': lon, 'mode': mode, 'lat': lat, }
    if top_cuisines:
        querystring['top_cuisines'] = top_cuisines
    if cuisine:
        querystring['cuisine'] = cuisine
    if fullmenu:
        querystring['fullmenu'] = fullmenu
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "documenu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_restaurants_geo(lon: int, distance: int, lat: int, top_cuisines: bool=None, size: int=30, page: int=2, fullmenu: bool=None, cuisine: str='Italian', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Function using location and search radius. Returns list of restaurants sorted by closest"
    lon: Longitude of the search area	
        distance: Search Area radius (in miles)	
        lat: Latitude of search area	
        top_cuisines: Show only Cuisines In Search Area	
        size: Number of Results To Return
        page: Page Through Results	
        fullmenu: Include full menus
        cuisine: Filter by cuisine 
        
    """
    url = f"https://documenu.p.rapidapi.com/restaurants/search/geo"
    querystring = {'lon': lon, 'distance': distance, 'lat': lat, }
    if top_cuisines:
        querystring['top_cuisines'] = top_cuisines
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    if fullmenu:
        querystring['fullmenu'] = fullmenu
    if cuisine:
        querystring['cuisine'] = cuisine
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "documenu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def restaurants_by_zip_code(zip_code: int, top_cuisines: bool=None, cuisine: str='Italian', fullmenu: bool=None, page: int=2, size: int=30, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Function Returns Restaurants By Zip Code"
    top_cuisines: Show only Cuisines In Search Area	
        cuisine: Cuisine To Search For	
        fullmenu: Include full menus
        page: Page Through Results	
        size: Size of Results To Return
        
    """
    url = f"https://documenu.p.rapidapi.com/restaurants/zip_code/{zip_code}"
    querystring = {}
    if top_cuisines:
        querystring['top_cuisines'] = top_cuisines
    if cuisine:
        querystring['cuisine'] = cuisine
    if fullmenu:
        querystring['fullmenu'] = fullmenu
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "documenu.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

