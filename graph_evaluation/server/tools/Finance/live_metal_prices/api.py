import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def latest_in_chosen_currency(requestedsymbols: str, requestedcurrency: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Select what you need for a smaller payload! Real-time Gold, Silver, Palladium, Platinum and 160+ currency rates based on selected Currency"
    
    """
    url = f"https://live-metal-prices.p.rapidapi.com/v1/latest/{requestedsymbols}/{requestedcurrency}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-metal-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_selected_metals_in_selected_currency_in_grams(requestedunitofweight: str, requestedcurrency: str, requestedsymbols: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Select what you need for a smaller payload! Real-time Gold, Silver, Palladium, Platinum and 160+ currency rates based on selected Currency"
    
    """
    url = f"https://live-metal-prices.p.rapidapi.com/v1/latest/{requestedsymbols}/{requestedcurrency}/{requestedunitofweight}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-metal-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_retrieve_xau_xag_pa_pl_eur_gbp_usd(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Real-time Gold, Silver, Palladium and Platinum prices delivered in USD, GBP and EUR."
    
    """
    url = f"https://live-metal-prices.p.rapidapi.com/v1/latest"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-metal-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def latest_retrieve_selected_160_symbols(requestedsymbols: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Select what you need for a smaller payload! Real-time Gold, Silver, Palladium, and Platinum provided in 160+ currencies including USD, GBP and EUR."
    
    """
    url = f"https://live-metal-prices.p.rapidapi.com/v1/latest/{requestedsymbols}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "live-metal-prices.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

