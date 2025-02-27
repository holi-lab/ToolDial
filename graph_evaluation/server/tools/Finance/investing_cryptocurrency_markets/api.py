import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def coins_list_pairs(lang_id: int=1, time_utc_offset: int=28800, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List available crypto pairs"
    lang_id: The value of all&#95;langs/lang&#95;ID returned in .../get-meta-data endpoint
        time_utc_offset: UTC value in seconds, for example : utc+8 -> 8 * 60 * 60 = 28800
        
    """
    url = f"https://investing-cryptocurrency-markets.p.rapidapi.com/coins/list-pairs"
    querystring = {}
    if lang_id:
        querystring['lang_ID'] = lang_id
    if time_utc_offset:
        querystring['time_utc_offset'] = time_utc_offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "investing-cryptocurrency-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_get_news(pair_id: int, lang_id: int=1, page: int=1, time_utc_offset: int=28800, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get news relating to specific cryptocurrency"
    pair_id: Value of pair&#95;id field returned in coins/list, coins/search, coins/list-pairs, etc...
        lang_id: The value of all&#95;langs/lang&#95;ID returned in .../get-meta-data endpoint
        page: For paging purpose
        time_utc_offset: UTC value in seconds, for example : utc+8 -> 8 * 60 * 60 = 28800
        
    """
    url = f"https://investing-cryptocurrency-markets.p.rapidapi.com/coins/get-news"
    querystring = {'pair_ID': pair_id, }
    if lang_id:
        querystring['lang_ID'] = lang_id
    if page:
        querystring['page'] = page
    if time_utc_offset:
        querystring['time_utc_offset'] = time_utc_offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "investing-cryptocurrency-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_get_technical(pair_id: int, time_utc_offset: int=28800, lang_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get technical information of specific cryptocurrency"
    pair_id: Value of pair&#95;id field returned in coins/list, coins/search, coins/list-pairs, etc...
        time_utc_offset: UTC value in seconds, for example : utc+8 -> 8 * 60 * 60 = 28800
        lang_id: The value of all&#95;langs/lang&#95;ID returned in .../get-meta-data endpoint
        
    """
    url = f"https://investing-cryptocurrency-markets.p.rapidapi.com/coins/get-technical"
    querystring = {'pair_ID': pair_id, }
    if time_utc_offset:
        querystring['time_utc_offset'] = time_utc_offset
    if lang_id:
        querystring['lang_ID'] = lang_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "investing-cryptocurrency-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_search(string: str, time_utc_offset: int=28800, lang_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search for available cryptocurrencies relating to a word, name, etc..."
    string: A word or name of any currencies
        time_utc_offset: UTC value in seconds, for example : utc+8 -> 8 * 60 * 60 = 28800
        lang_id: The value of all&#95;langs/lang&#95;ID returned in .../get-meta-data endpoint
        
    """
    url = f"https://investing-cryptocurrency-markets.p.rapidapi.com/coins/search"
    querystring = {'string': string, }
    if time_utc_offset:
        querystring['time_utc_offset'] = time_utc_offset
    if lang_id:
        querystring['lang_ID'] = lang_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "investing-cryptocurrency-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_get_analysis(pair_id: int, time_utc_offset: int=28800, lang_id: int=1, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get analysis opinion from expert for specific cryptocurrency"
    pair_id: Value of pair&#95;id field returned in coins/list, coins/search, coins/list-pairs, etc...
        time_utc_offset: UTC value in seconds, for example : utc+8 -> 8 * 60 * 60 = 28800
        lang_id: The value of all&#95;langs/lang&#95;ID returned in .../get-meta-data endpoint
        page: For paging purpose
        
    """
    url = f"https://investing-cryptocurrency-markets.p.rapidapi.com/coins/get-analysis"
    querystring = {'pair_ID': pair_id, }
    if time_utc_offset:
        querystring['time_utc_offset'] = time_utc_offset
    if lang_id:
        querystring['lang_ID'] = lang_id
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "investing-cryptocurrency-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_get_fullsize_chart(pair_id: int, pair_interval: int=900, lang_id: int=1, time_utc_offset: int=28800, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get fullsize chart of specific cryptocurrency"
    pair_id: Value of pair&#95;id field returned in coins/list, coins/search, coins/list-pairs, etc...
        pair_interval: One of the following is allowed 60 (1min)|300 (5min)|900 (15min)|1800 (30min)|3600 (1h)|18000 (5h)|86400 (1d)|week (1w) |month (1m)
        lang_id: The value of all&#95;langs/lang&#95;ID returned in .../get-meta-data endpoint
        time_utc_offset: UTC value in seconds, for example : utc+8 -> 8 * 60 * 60 = 28800
        
    """
    url = f"https://investing-cryptocurrency-markets.p.rapidapi.com/coins/get-fullsize-chart"
    querystring = {'pair_ID': pair_id, }
    if pair_interval:
        querystring['pair_interval'] = pair_interval
    if lang_id:
        querystring['lang_ID'] = lang_id
    if time_utc_offset:
        querystring['time_utc_offset'] = time_utc_offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "investing-cryptocurrency-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_get_historical_data(date_from: str, date_to: str, pair_id: int, lang_id: int=1, time_utc_offset: int=28800, interval: str='day', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get historical data of specific cryptocurrency"
    date_from: Date in ddmmyyyy format
        date_to: Date in ddmmyyyy format
        pair_id: Value of pair&#95;id field returned in coins/list, coins/search, coins/list-pairs, etc...
        lang_id: The value of all&#95;langs/lang&#95;ID returned in .../get-meta-data endpoint
        time_utc_offset: UTC value in seconds, for example : utc+8 -> 8 * 60 * 60 = 28800
        interval: One of the following is allowed  day|week|month
        
    """
    url = f"https://investing-cryptocurrency-markets.p.rapidapi.com/coins/get-historical-data"
    querystring = {'date_from': date_from, 'date_to': date_to, 'pair_ID': pair_id, }
    if lang_id:
        querystring['lang_ID'] = lang_id
    if time_utc_offset:
        querystring['time_utc_offset'] = time_utc_offset
    if interval:
        querystring['interval'] = interval
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "investing-cryptocurrency-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_get_markets(cur2: int, pair_id: int, time_utc_offset: int=28800, lang_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get market information of specific cryptocurrency in specified currency"
    cur2: The value of edition&#95;currency&#95;id returned in .../get-meta-data endpoint
        pair_id: Value of pair&#95;id field returned in coins/list, coins/search, coins/list-pairs, etc...
        time_utc_offset: UTC value in seconds, for example : utc+8 -> 8 * 60 * 60 = 28800
        lang_id: The value of all&#95;langs/lang&#95;ID returned in .../get-meta-data endpoint
        
    """
    url = f"https://investing-cryptocurrency-markets.p.rapidapi.com/coins/get-markets"
    querystring = {'cur2': cur2, 'pair_ID': pair_id, }
    if time_utc_offset:
        querystring['time_utc_offset'] = time_utc_offset
    if lang_id:
        querystring['lang_ID'] = lang_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "investing-cryptocurrency-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_get_overview(pair_id: int, time_utc_offset: int=28800, lang_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get overview information of specific cryptocurrency"
    pair_id: Value of pair&#95;id field returned in coins/list, coins/search, coins/list-pairs, etc...
        time_utc_offset: UTC value in seconds, for example : utc+8 -> 8 * 60 * 60 = 28800
        lang_id: The value of all&#95;langs/lang&#95;ID returned in .../get-meta-data endpoint
        
    """
    url = f"https://investing-cryptocurrency-markets.p.rapidapi.com/coins/get-overview"
    querystring = {'pair_ID': pair_id, }
    if time_utc_offset:
        querystring['time_utc_offset'] = time_utc_offset
    if lang_id:
        querystring['lang_ID'] = lang_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "investing-cryptocurrency-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_meta_data(locale_info: str, lang_id: int=1, time_utc_offset: int=28800, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get init meta data"
    locale_info: The language code
        lang_id: The value of all&#95;langs/lang&#95;ID returned in .../get-meta-data endpoint
        time_utc_offset: UTC value in seconds, for example : utc+8 -> 8 * 60 * 60 = 28800
        
    """
    url = f"https://investing-cryptocurrency-markets.p.rapidapi.com/get-meta-data"
    querystring = {'locale_info': locale_info, }
    if lang_id:
        querystring['lang_ID'] = lang_id
    if time_utc_offset:
        querystring['time_utc_offset'] = time_utc_offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "investing-cryptocurrency-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_ico_calendar(category: str=None, time_utc_offset: int=28800, tabname: str='ongoing', sort: str='related_days', lang_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get ICO calendar"
    category: Check for suitable value of icoData/categories returned right in this endpoint. Separated by comma for multiple options. For example : &#95;ico&#95;cat&#95;gaming,&#95;ico&#95;cat&#95;ecomm,&#95;ico&#95;cat&#95;finance,&#95;ico&#95;cat&#95;healthcare
        time_utc_offset: UTC value in seconds, for example : utc+8 -> 8 * 60 * 60 = 28800
        tabname: One of the following : upcoming|completed|ongoing
        sort: One of the following is allowed related&#95;days | name | funds&#95;raised | completed
        lang_id: The value of all&#95;langs/lang&#95;ID returned in .../get-meta-data endpoint
        
    """
    url = f"https://investing-cryptocurrency-markets.p.rapidapi.com/get-ico-calendar"
    querystring = {}
    if category:
        querystring['category'] = category
    if time_utc_offset:
        querystring['time_utc_offset'] = time_utc_offset
    if tabname:
        querystring['tabname'] = tabname
    if sort:
        querystring['sort'] = sort
    if lang_id:
        querystring['lang_ID'] = lang_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "investing-cryptocurrency-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_get_brief_chart(pair_id: int, lang_id: int=1, range: str='p', time_utc_offset: int=28800, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get brief information chart of specific cryptocurrency"
    pair_id: Value of pair&#95;id field returned in coins/list, coins/search, coins/list-pairs, etc...
        lang_id: The value of all&#95;langs/lang&#95;ID returned in .../get-meta-data endpoint
        range: One of the following is allowed d|w|1m|1y|5y|max
        time_utc_offset: UTC value in seconds, for example : utc+8 -> 8 * 60 * 60 = 28800
        
    """
    url = f"https://investing-cryptocurrency-markets.p.rapidapi.com/coins/get-brief-chart"
    querystring = {'pair_ID': pair_id, }
    if lang_id:
        querystring['lang_ID'] = lang_id
    if range:
        querystring['range'] = range
    if time_utc_offset:
        querystring['time_utc_offset'] = time_utc_offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "investing-cryptocurrency-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currencies_list(time_utc_offset: int=28800, lang_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available currencies"
    time_utc_offset: UTC value in seconds, for example : utc+8 -> 8 * 60 * 60 = 28800
        lang_id: The value of all&#95;langs/lang&#95;ID returned in .../get-meta-data endpoint
        
    """
    url = f"https://investing-cryptocurrency-markets.p.rapidapi.com/currencies/list"
    querystring = {}
    if time_utc_offset:
        querystring['time_utc_offset'] = time_utc_offset
    if lang_id:
        querystring['lang_ID'] = lang_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "investing-cryptocurrency-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_list(edition_currency_id: int, total_volume_min: int=None, chg_24h_min: int=None, lang_id: int=1, total_volume_max: int=None, chg_7d_max: int=None, time_utc_offset: int=28800, chg_7d_min: int=None, market_cap_max: int=None, market_cap_min: int=None, chg_24h_max: int=None, volume_24h_max: int=None, volume_24h_min: int=None, sort: str='PERC1D_DN', page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all available cryptocurrencies"
    edition_currency_id: The value of edition&#95;currency&#95;id returned in .../get-meta-data endpoint
        total_volume_min: Check rangeFilters/total&#95;volume/steps field returned right in this endpoint for available min and max value 
        chg_24h_min: Check rangeFilters/chg&#95;24h/steps field returned right in this endpoint for available min and max value 
        lang_id: The value of all&#95;langs/lang&#95;ID returned in .../get-meta-data endpoint
        total_volume_max: Check rangeFilters/total_volume/steps field returned right in this endpoint for available min and max value 
        chg_7d_max: Check rangeFilters/chg&#95;7d/steps field returned right in this endpoint for available min and max value 
        time_utc_offset: UTC value in seconds, for example : utc+8 -> 8 * 60 * 60 = 28800
        chg_7d_min: Check rangeFilters/chg&#95;7d/steps field returned right in this endpoint for available min and max value 
        market_cap_max: Check rangeFilters/market&#95;cap/steps field returned right in this endpoint for available min and max value
        market_cap_min: Check rangeFilters/market&#95;cap/steps field returned right in this endpoint for available min and max value
        chg_24h_max: Check rangeFilters/chg&#95;24h/steps field returned right in this endpoint for available min and max value 
        volume_24h_max: Check rangeFilters/volume&#95;24h/steps field returned right in this endpoint for available min and max value 
        volume_24h_min: Check rangeFilters/volume&#95;24h/steps field returned right in this endpoint for available min and max value 
        sort: One of the following is allowed NAME&#95;UP (Name) | PERC1D&#95;DN (Chg 24h) | PERC7D&#95;DN (Chg 7D) | MARKETCAP&#95;DN (Market Cap) | VOLUME24&#95;DN (Vol 24h) | TOTAL&#95;VOLUME&#95;DN (Total vol)
        page: For paging purpose
        
    """
    url = f"https://investing-cryptocurrency-markets.p.rapidapi.com/coins/list"
    querystring = {'edition_currency_id': edition_currency_id, }
    if total_volume_min:
        querystring['total_volume_min'] = total_volume_min
    if chg_24h_min:
        querystring['chg_24h_min'] = chg_24h_min
    if lang_id:
        querystring['lang_ID'] = lang_id
    if total_volume_max:
        querystring['total_volume_max'] = total_volume_max
    if chg_7d_max:
        querystring['chg_7d_max'] = chg_7d_max
    if time_utc_offset:
        querystring['time_utc_offset'] = time_utc_offset
    if chg_7d_min:
        querystring['chg_7d_min'] = chg_7d_min
    if market_cap_max:
        querystring['market_cap_max'] = market_cap_max
    if market_cap_min:
        querystring['market_cap_min'] = market_cap_min
    if chg_24h_max:
        querystring['chg_24h_max'] = chg_24h_max
    if volume_24h_max:
        querystring['volume_24h_max'] = volume_24h_max
    if volume_24h_min:
        querystring['volume_24h_min'] = volume_24h_min
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "investing-cryptocurrency-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def currencies_get_rate(fromcurrency: int, tocurrency: int, time_utc_offset: int=28800, lang_id: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get exchange rate between two different currencies"
    fromcurrency: Value of currency&#95;ID field returned in currencies/list endpoint
        tocurrency: Value of currency&#95;ID field returned in currencies/list endpoint
        time_utc_offset: UTC value in seconds, for example : utc+8 -> 8 * 60 * 60 = 28800
        lang_id: The value of all&#95;langs/lang&#95;ID returned in .../get-meta-data endpoint
        
    """
    url = f"https://investing-cryptocurrency-markets.p.rapidapi.com/currencies/get-rate"
    querystring = {'fromCurrency': fromcurrency, 'toCurrency': tocurrency, }
    if time_utc_offset:
        querystring['time_utc_offset'] = time_utc_offset
    if lang_id:
        querystring['lang_ID'] = lang_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "investing-cryptocurrency-markets.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

