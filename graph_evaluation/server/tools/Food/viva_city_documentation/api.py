import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def menudetails(is_id: str, lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get menu details using KV (need resync to get latest data) and filter dynamic multi language data based on query params."
    id: You can get the list of the menu ids from ```/vebue-i8n/menus/{id}/details```  with the key of  **"menu_item"**
        
    """
    url = f"https://viva-city-documentation.p.rapidapi.com/venue-i18n/menus/{is_id}/details"
    querystring = {'lang': lang, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "viva-city-documentation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def venuedetails(is_id: str, lang: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get venue details using KV (need resync to get latest data) and filter dynamic multi language data based on query params."
    
    """
    url = f"https://viva-city-documentation.p.rapidapi.com/venue-i18n/venues/{is_id}/details"
    querystring = {'lang': lang, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "viva-city-documentation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

