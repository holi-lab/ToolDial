import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def market_trends(t: str, gl: str='US', s: str=None, hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Explore market trends"
    t: Tab name. One of these:

- `indexes`
- `most-active`
- `gainers`
- `losers`
- `climate-leaders`
- `cryptocurrencies`
- `currencies`
        s: Subtab name. If you selected the `indexes` tab, you can choose one of the following, leave empty to get all.

- `americas`
- `europe-middle-east-africa`
- `asia-pacific`
        
    """
    url = f"https://joj-finance.p.rapidapi.com/market-trends/"
    querystring = {'t': t, }
    if gl:
        querystring['gl'] = gl
    if s:
        querystring['s'] = s
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joj-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ticker(t: str, gl: str='US', hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get stats, price, charts, news and information about indexes, stocks, ETFs, cryptocurrencies, currencies and more"
    t: Ticker symbol. For example:

- `ABNB:NASDAQ`
- `.DJI:INDEXDJX`
- `EUR-USD`
- `BTC-USD`
        
    """
    url = f"https://joj-finance.p.rapidapi.com/ticker/"
    querystring = {'t': t, }
    if gl:
        querystring['gl'] = gl
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joj-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search(q: str, gl: str='US', hl: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for indexes, stocks, ETFs, cryptocurrencies, currencies and more"
    q: Search query.
        
    """
    url = f"https://joj-finance.p.rapidapi.com/search/"
    querystring = {'q': q, }
    if gl:
        querystring['gl'] = gl
    if hl:
        querystring['hl'] = hl
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "joj-finance.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

