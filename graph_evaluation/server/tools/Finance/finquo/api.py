import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def top_traded_stocks(limit: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Most traded/growing securities."
    
    """
    url = f"https://finquo5.p.rapidapi.com/p2/topTraded"
    querystring = {}
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finquo5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def custom_strategy(email: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Order Custom Strategy"
    email: Order the development of an individual strategy
        
    """
    url = f"https://finquo5.p.rapidapi.com/strategy/custom"
    querystring = {}
    if email:
        querystring['email'] = email
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finquo5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subscribe_for_news(email: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Subscribe to the newsletter that we collect from all over the Internet with analytics and price impact forecasts"
    
    """
    url = f"https://finquo5.p.rapidapi.com/p4/subscribe"
    querystring = {'email': email, 'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finquo5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def real_time_price_info(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Real Time Price"
    symbol: Symbol can be an array. symbol=[AAPL,MDB]
        
    """
    url = f"https://finquo5.p.rapidapi.com/p2/price"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finquo5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def real_time_last_candle_prices(timeframe: str, symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Real Time Last Candle Prices. 
		Format Candle [DateTime UTC+0, Open, High, Low, Close, Volume]"
    
    """
    url = f"https://finquo5.p.rapidapi.com/p2/hist?live=1"
    querystring = {'timeframe': timeframe, 'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finquo5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quotes_pre_post_rth_market_real_time(live: str, symbol: str, timeframe: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All quotes for all the time + Unfinished Candle"
    
    """
    url = f"https://finquo5.p.rapidapi.com/p2/hist"
    querystring = {'live': live, 'symbol': symbol, 'timeframe': timeframe, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finquo5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quotes_regular_trading_hours_history(timeframe: str, symbol: str, to: str=None, is_from: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get historical data Stock Quotes, just only for Regular Trading Hours: American/New York TZ: 9:30 am - 4pm.
		
		Candle [DateTime UTC, Open, High, Low, Close, Volume]"
    
    """
    url = f"https://finquo5.p.rapidapi.com/p1/hist"
    querystring = {'timeframe': timeframe, 'symbol': symbol, }
    if to:
        querystring['to'] = to
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finquo5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def holiday_and_weekend_calendar(country: str, year: str=None, month: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Dates of all holidays and weekends in the past and future"
    
    """
    url = f"https://finquo5.p.rapidapi.com/p4/weekends"
    querystring = {'country': country, }
    if year:
        querystring['year'] = year
    if month:
        querystring['month'] = month
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finquo5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def alerts(email: str, price_event: str, symbol: str, timeframe: str, price_value: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Subscribe for alerts when price across it."
    timeframe: 1m, 5m, 30m, 60m, 1d. If you need other timeframes - contact to us
        
    """
    url = f"https://finquo5.p.rapidapi.com/p4/alerts"
    querystring = {'email': email, 'price_event': price_event, 'symbol': symbol, 'timeframe': timeframe, }
    if price_value:
        querystring['price_value'] = price_value
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finquo5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_symbol(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can find the symbol you need to retrieve data for by using our search functionality. It allows you to search for symbols based on keywords or terms. Once you find the desired symbol, you will obtain its identifier, which you can then use to request data specifically for that symbol. This convenient search feature simplifies the process of accessing the data you require, ensuring that you can quickly and easily retrieve the information you need."
    
    """
    url = f"https://finquo5.p.rapidapi.com/search"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finquo5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trading_strategy(email: str, strategy: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Subscribe to receive information about entry points for our authoring strategies. We can also develop your individual strategy for you."
    email: Enter your email where we send the signal
        
    """
    url = f"https://finquo5.p.rapidapi.com/p4/strategy"
    querystring = {'email': email, 'strategy': strategy, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finquo5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_news(symbol: str, q: str='stock+share', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get news by symbol"
    
    """
    url = f"https://finquo5.p.rapidapi.com/p4/news"
    querystring = {'symbol': symbol, }
    if q:
        querystring['q'] = q
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finquo5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quotes_regular_trading_hours_real_time(timeframe: str, symbol: str, to: str=None, live: str=None, is_from: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Quotes Regular Trading Hours for current trading session with Unfinished Candle"
    
    """
    url = f"https://finquo5.p.rapidapi.com/p2/hist?session=1"
    querystring = {'timeframe': timeframe, 'symbol': symbol, }
    if to:
        querystring['to'] = to
    if live:
        querystring['live'] = live
    if is_from:
        querystring['from'] = is_from
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finquo5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gateway_status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check Server Status"
    
    """
    url = f"https://finquo5.p.rapidapi.com/status"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finquo5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

