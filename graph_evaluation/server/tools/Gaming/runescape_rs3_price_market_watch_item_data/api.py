import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def rs3_items_search_any_field(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search category title, change category to field name to search
		search from fields available replace category with field name and search_value=value with the search value.
		
		change item_id to an available field and then the search value 
		e.g: /rs3_items/search_list/item_id/?search_value=3022"
    
    """
    url = f"https://runescape-rs3-price-market-watch-item-data.p.rapidapi.com/rs3_items/search_list/item_id/?search_value=3022"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "runescape-rs3-price-market-watch-item-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rs3_prices_search_any_field(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search category title, change category to field name to search search from fields available replace category with field name and search_value=value with the search value. RS3 prices
		
		change the item_id to an available field and then the search_value = to the .. search value required.  
		e.g: /rs3_prices/search_list/item_id/?search_value=11726"
    
    """
    url = f"https://runescape-rs3-price-market-watch-item-data.p.rapidapi.com/rs3_prices/search_list/item_id/?search_value=11726"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "runescape-rs3-price-market-watch-item-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rs3_prices_list_all_fields_from_multiple_items(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pass in multiple items into this search and return results from RS3 prices. max = number items to return."
    
    """
    url = f"https://runescape-rs3-price-market-watch-item-data.p.rapidapi.com/rs3_prices/search_distinct/{category}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "runescape-rs3-price-market-watch-item-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rs3_prices_distinct_list_of_any_field_values(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "for example if you want to get a distinct list of categories that exist from RS3 prices. this should show a distinct list in json array."
    
    """
    url = f"https://runescape-rs3-price-market-watch-item-data.p.rapidapi.com/rs3_prices/search_distinct/{category}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "runescape-rs3-price-market-watch-item-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rs3_prices_latest_50_100_items(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "show the latest items from the range starting at 50 and showing 100 items from RS3 prices"
    
    """
    url = f"https://runescape-rs3-price-market-watch-item-data.p.rapidapi.com/rs3_prices/latest/50/100/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "runescape-rs3-price-market-watch-item-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rs3_prices_latest_50_items(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "show the latest 50 items from RS3 prices"
    
    """
    url = f"https://runescape-rs3-price-market-watch-item-data.p.rapidapi.com/rs3_prices/latest/start/50/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "runescape-rs3-price-market-watch-item-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rs3_prices_list_items_fields_available(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all available fields in rs3_prices"
    
    """
    url = f"https://runescape-rs3-price-market-watch-item-data.p.rapidapi.com/rs3_prices/list_fields/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "runescape-rs3-price-market-watch-item-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rs3_items_distinct_list_of_any_field_values(category: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "for example if you want to get a distinct list of categories that exist. this should show a distinct list in json array."
    
    """
    url = f"https://runescape-rs3-price-market-watch-item-data.p.rapidapi.com/rs3_items/search_distinct/{category}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "runescape-rs3-price-market-watch-item-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rs3_items_latest_50_items(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "show the latest 50 items"
    
    """
    url = f"https://runescape-rs3-price-market-watch-item-data.p.rapidapi.com/rs3_items/latest/start/50/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "runescape-rs3-price-market-watch-item-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rs3_items_list_items_fields_available(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists all available fields in rs3_items"
    
    """
    url = f"https://runescape-rs3-price-market-watch-item-data.p.rapidapi.com/rs3_items/list_fields/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "runescape-rs3-price-market-watch-item-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rs3_items_latest_50_100_items(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "show the latest items from the range starting at 50 and showing 100 items from RS3 items"
    
    """
    url = f"https://runescape-rs3-price-market-watch-item-data.p.rapidapi.com/rs3_items/latest/50/100/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "runescape-rs3-price-market-watch-item-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rs3_items_list_all_fields_from_multiple_items(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Pass in multiple items into this search and return results. max = number items to return."
    
    """
    url = f"https://runescape-rs3-price-market-watch-item-data.p.rapidapi.com/rs3_items/search_multi/?category=value1&title=value2&max=50"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "runescape-rs3-price-market-watch-item-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

