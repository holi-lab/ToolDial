import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def auto_complete_deprecated(query: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Auto suggestion by input name or quote.
		* This endpoint is deprecated"
    query: Symbol or company name
        
    """
    url = f"https://fidelity-investments.p.rapidapi.com/auto-complete"
    querystring = {'query': query, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fidelity-investments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_auto_complete_deprecated(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Auto suggestion by input name or quote"
    q: Symbol or company name
        
    """
    url = f"https://fidelity-investments.p.rapidapi.com/v2/auto-complete"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fidelity-investments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v3_auto_complete(q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Auto suggestion by input name or quote"
    q: Symbol or company name
        
    """
    url = f"https://fidelity-investments.p.rapidapi.com/v3/auto-complete"
    querystring = {'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fidelity-investments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quotes_get_chart(symbol: str, startdate: str, enddate: str, intraday: str='Y', granularity: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information to draw chart"
    symbol: Separated by comma for multiple symbols, support up to 3 symbols at a time
        startdate: Date format must be strictly follow yyyy/MM/dd-HH:mm:ss
        enddate: Date format must be strictly follow yyyy/MM/dd-HH:mm:ss
        intraday: Y or N
        granularity: From 1 to 6, use with intraday to specify day or month
        
    """
    url = f"https://fidelity-investments.p.rapidapi.com/quotes/get-chart"
    querystring = {'symbol': symbol, 'startDate': startdate, 'endDate': enddate, }
    if intraday:
        querystring['intraday'] = intraday
    if granularity:
        querystring['granularity'] = granularity
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fidelity-investments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_international(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get international markets information"
    
    """
    url = f"https://fidelity-investments.p.rapidapi.com/market/get-international"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fidelity-investments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_get_details(resid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news details"
    resid: The value of resId field returned in .../news/list-top endpoint, such as : 202003011902RTRSNEWSCOMBINED_KBN20O2GM-OUSBS_1
        
    """
    url = f"https://fidelity-investments.p.rapidapi.com/news/get-details"
    querystring = {'resId': resid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fidelity-investments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quotes_get_details(symbols: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get quote information"
    symbols: Separated by comma to query multiple symbols 
        
    """
    url = f"https://fidelity-investments.p.rapidapi.com/quotes/get-details"
    querystring = {'symbols': symbols, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fidelity-investments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_orders(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get orders by Fidelity customers"
    
    """
    url = f"https://fidelity-investments.p.rapidapi.com/market/get-orders"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fidelity-investments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def news_list_top(symbol: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List top news from all supported area"
    symbol: The symbol of quote, market, etc..., such as : IMRN. Only one is allowed at a time
        
    """
    url = f"https://fidelity-investments.p.rapidapi.com/news/list-top"
    querystring = {}
    if symbol:
        querystring['symbol'] = symbol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fidelity-investments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_movers(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market movers which are most actives, top gainers, top losers"
    
    """
    url = f"https://fidelity-investments.p.rapidapi.com/market/get-movers"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fidelity-investments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market_get_sectors(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get sectors performance"
    
    """
    url = f"https://fidelity-investments.p.rapidapi.com/market/get-sectors"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fidelity-investments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def quotes_get_mashup(symbol: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get additional information for specific quote, market"
    symbol: Only one symbol is allowed
        
    """
    url = f"https://fidelity-investments.p.rapidapi.com/quotes/get-mashup"
    querystring = {'symbol': symbol, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "fidelity-investments.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

