import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def exchange_info(exchange: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Accepts an exchange name as an input, returns a list of exchanges where the coin is traded, and prices for this coin"
    
    """
    url = f"https://new-cryptocurrencies-listings.p.rapidapi.com/exchange_info"
    querystring = {'exchange': exchange, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "new-cryptocurrencies-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news(limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns latest news from all supported exchanges"
    
    """
    url = f"https://new-cryptocurrencies-listings.p.rapidapi.com/news"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "new-cryptocurrencies-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_pairs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns upcoming currency pair exits on all supported exchanges"
    
    """
    url = f"https://new-cryptocurrencies-listings.p.rapidapi.com/new_pairs"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "new-cryptocurrencies-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def asset_info(tiker: str, limit: str='10', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Accepts a coin ticker as an input, returns a list of exchanges where the coin is traded and prices for this coin
		
		**Note: default value for limit is 10**"
    
    """
    url = f"https://new-cryptocurrencies-listings.p.rapidapi.com/asset_info"
    querystring = {'tiker': tiker, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "new-cryptocurrencies-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def set_alert(email: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can send your email address and for new listings, it will send notifications, if there are no errors it will return status: ok and send a test letter to the email address. Later you will receive an email if new coins start listing at supported exchanges"
    
    """
    url = f"https://new-cryptocurrencies-listings.p.rapidapi.com/set_alert"
    querystring = {'email': email, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "new-cryptocurrencies-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def supported_exchanges(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a list of exchanges from which we receive data"
    
    """
    url = f"https://new-cryptocurrencies-listings.p.rapidapi.com/supported_exchanges"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "new-cryptocurrencies-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def send_email(exchange: str='Binance', date: str='2021-08-23 02:30', email: str='example@example.com', coin: str='BTC', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can send an email as notification about a new listing to be able to buy it before the price rase"
    
    """
    url = f"https://new-cryptocurrencies-listings.p.rapidapi.com/send_email"
    querystring = {}
    if exchange:
        querystring['exchange'] = exchange
    if date:
        querystring['date'] = date
    if email:
        querystring['email'] = email
    if coin:
        querystring['coin'] = coin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "new-cryptocurrencies-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def new_listings(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns new listings on biggest exchanges(now just Binance, but I'll fix it later)
		- status: "ok" or "error"
		- anounsments: list of crypto announcements 
		-- date: date of announcement at the exchange
		-- text: text of the announcement from the exchange
		-- title: title of the announcement from the exchange
		-- topic: -
		-- ticker: ticker of new currency
		-- news_url: URL of the announcement
		-- news_type: -
		-- sentiment: -
		-- source_name: name of the exchange
		
		**Note**: updates every 10 minutes"
    
    """
    url = f"https://new-cryptocurrencies-listings.p.rapidapi.com/new_listings"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "new-cryptocurrencies-listings.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

