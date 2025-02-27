import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_buy_sell_informations_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "From a given ID, retrieve the latest Buy/Sell orders placed on the Steam Store."
    id: Retrieve this ID from the Get Items IDs by query endpoint
        
    """
    url = f"https://steam-market-and-store.p.rapidapi.com/get_orders_hist/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-market-and-store.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_latest_activity_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given a specific ID, retrieve the last activity dealed with it. Basic information about the user performing the activity is also provided."
    id: Retrieve this ID from the Get Items IDs by query endpoint
        
    """
    url = f"https://steam-market-and-store.p.rapidapi.com/get_last_act/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-market-and-store.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_price_history_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For a given ID, get the price history together with the number of items sold."
    id: Retrieve this ID from the Get Items IDs by query endpoint
        
    """
    url = f"https://steam-market-and-store.p.rapidapi.com/get_price_history/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-market-and-store.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_items_ids_by_query(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter a (text) query and retrieve the IDS associated to your items. This ID will be needed in order to explore market information."
    query: Text Query
        
    """
    url = f"https://steam-market-and-store.p.rapidapi.com/get_ids/{query}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "steam-market-and-store.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

