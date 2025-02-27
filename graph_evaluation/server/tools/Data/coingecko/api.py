import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def exchanges_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get exchange volume in BTC and top 100 tickers only"
    id: pass the exchange id (can be obtained from /exchanges/list) eg. binance
        
    """
    url = f"https://coingecko.p.rapidapi.com/exchanges/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events(country_code: str=None, type: str=None, page: str=None, from_date: str=None, to_date: str=None, upcoming_events_only: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get events, paginated by 100"
    country_code: country_code of event (eg. ‘US’). use /api/v3/events/countries for list of country_codes
        type: type of event (eg. ‘Conference’). use /api/v3/events/types for list of types
        page: page of results (paginated by 100)
        from_date: lists events after this date yyyy-mm-dd
        to_date: lists events before this date yyyy-mm-dd (set upcoming_events_only to false if fetching past events)
        upcoming_events_only: lists only upcoming events.  true, false
        
    """
    url = f"https://coingecko.p.rapidapi.com/events"
    querystring = {}
    if country_code:
        querystring['country_code'] = country_code
    if type:
        querystring['type'] = type
    if page:
        querystring['page'] = page
    if from_date:
        querystring['from_date'] = from_date
    if to_date:
        querystring['to_date'] = to_date
    if upcoming_events_only:
        querystring['upcoming_events_only'] = upcoming_events_only
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_countries(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of event countries"
    
    """
    url = f"https://coingecko.p.rapidapi.com/events/countries"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def global(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get cryptocurrency global data"
    
    """
    url = f"https://coingecko.p.rapidapi.com/global"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def events_types(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of events types"
    
    """
    url = f"https://coingecko.p.rapidapi.com/events/types"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchanges(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all exchanges"
    
    """
    url = f"https://coingecko.p.rapidapi.com/exchanges"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_markets(vs_currency: str, price_change_percentage: str=None, page: str='1', sparkline: bool=None, per_page: str='100', ids: str=None, order: str='market_cap_desc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this to obtain all the coins market data (price, market cap, volume)"
    vs_currency: The target currency of market data (usd, eur, jpy, etc.)
        price_change_percentage: Include price change percentage in 1h, 24h, 7d, 14d, 30d, 200d, 1y (eg. '1h,24h,7d' comma-separated, invalid values will be discarded)
        page: Page through resuts
        sparkline: Include sparkline 7 days data (eg. true, false)
        per_page: Total results per page. Between 1 to 250
        ids: The ids of the coin, comma separated crytocurrency symbols (base). refers to /coins/list. When left empty, returns numbers the coins observing the params limit and start
        order: sort results by field. valid values: market_cap_desc, gecko_desc, gecko_asc, market_cap_asc, market_cap_desc, volume_asc, volume_desc
        
    """
    url = f"https://coingecko.p.rapidapi.com/coins/markets"
    querystring = {'vs_currency': vs_currency, }
    if price_change_percentage:
        querystring['price_change_percentage'] = price_change_percentage
    if page:
        querystring['page'] = page
    if sparkline:
        querystring['sparkline'] = sparkline
    if per_page:
        querystring['per_page'] = per_page
    if ids:
        querystring['ids'] = ids
    if order:
        querystring['order'] = order
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ping(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check API server status"
    
    """
    url = f"https://coingecko.p.rapidapi.com/ping"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_id_tickers(is_id: str, exchange_ids: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " Get coin tickers (paginated to 100 items)"
    id: pass the coin id (can be obtained from /coins/list) eg. bitcoin
        exchange_ids: filter results by exchange_ids (ref: v3/exchanges/list)
        page: Page through results
        
    """
    url = f"https://coingecko.p.rapidapi.com/coins/{is_id}/tickers"
    querystring = {}
    if exchange_ids:
        querystring['exchange_ids'] = exchange_ids
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def simple_price(ids: str, vs_currencies: str, include_last_updated_at: str='False', include_market_cap: str='False', include_24hr_change: str='False', include_24hr_vol: str='False', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the current price of any cryptocurrencies in any other supported currencies that you need."
    ids: id of coins, comma-separated if querying more than 1 coin *refers to coins/list
        vs_currencies: vs_currency of coins, comma-separated if querying more than 1 vs_currency *refers to simple/supported_vs_currencies
        include_last_updated_at: true/false to include last_updated_at of price
        include_market_cap: true/false to include market_cap
        include_24hr_change: true/false to include 24hr_change
        include_24hr_vol: true/false to include 24hr_vol
        
    """
    url = f"https://coingecko.p.rapidapi.com/simple/price"
    querystring = {'ids': ids, 'vs_currencies': vs_currencies, }
    if include_last_updated_at:
        querystring['include_last_updated_at'] = include_last_updated_at
    if include_market_cap:
        querystring['include_market_cap'] = include_market_cap
    if include_24hr_change:
        querystring['include_24hr_change'] = include_24hr_change
    if include_24hr_vol:
        querystring['include_24hr_vol'] = include_24hr_vol
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_id_market_chart(vs_currency: str, days: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get historical market data include price, market cap, and 24h volume (granularity auto)"
    vs_currency: The target currency of market data (usd, eur, jpy, etc.)
        days: Data up to number of days ago (eg. 1,14,30,max)
        id: pass the coin id (can be obtained from /coins) eg. bitcoin
        
    """
    url = f"https://coingecko.p.rapidapi.com/coins/{is_id}/market_chart"
    querystring = {'vs_currency': vs_currency, 'days': days, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchanges_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all supported markets id and name (no pagination required)"
    
    """
    url = f"https://coingecko.p.rapidapi.com/exchanges/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_id(is_id: str, localization: str='true', tickers: bool=None, market_data: bool=None, community_data: bool=None, developer_data: bool=None, sparkline: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current data (name, price, market, ... including exchange tickers) for a coin"
    id: pass the coin id (can be obtained from /coins) eg. bitcoin
        localization: Include all localized languages in response (true/false)
        tickers: Include tickers data (true/false)
        market_data: Include market_data (true/false) 
        community_data: Include community_data data (true/false)
        developer_data: Include developer_data data (true/false)
        sparkline: Include sparkline 7 days data (eg. true, false)
        
    """
    url = f"https://coingecko.p.rapidapi.com/coins/{is_id}"
    querystring = {}
    if localization:
        querystring['localization'] = localization
    if tickers:
        querystring['tickers'] = tickers
    if market_data:
        querystring['market_data'] = market_data
    if community_data:
        querystring['community_data'] = community_data
    if developer_data:
        querystring['developer_data'] = developer_data
    if sparkline:
        querystring['sparkline'] = sparkline
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_id_contract_contract_address(contract_address: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get coin info from contract address"
    contract_address: Token’s contract address
        id: Asset platform (only ethereum is supported at this moment)
        
    """
    url = f"https://coingecko.p.rapidapi.com/coins/{is_id}/contract/{contract_address}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_id_status_updates(is_id: str, per_page: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get status updates for a given coin"
    id: pass the coin id (can be obtained from /coins) eg. bitcoin
        per_page: Total results per page
        page: Page through results
        
    """
    url = f"https://coingecko.p.rapidapi.com/coins/{is_id}/status_updates"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchange_rates(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get BTC-to-Currency exchange rates"
    
    """
    url = f"https://coingecko.p.rapidapi.com/exchange_rates"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_id_market_chart_range(is_from: str, vs_currency: str, to: str, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get historical market data include price, market cap, and 24h volume within a range of timestamp (granularity auto)"
    from: From date in UNIX Timestamp (eg. 1392577232)
        vs_currency: The target currency of market data (usd, eur, jpy, etc.)
        to: To date in UNIX Timestamp (eg. 1422577232)
        id: pass the coin id (can be obtained from /coins) eg. bitcoin
        
    """
    url = f"https://coingecko.p.rapidapi.com/coins/{is_id}/market_chart/range"
    querystring = {'from': is_from, 'vs_currency': vs_currency, 'to': to, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchanges_id_status_updates(is_id: str, per_page: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get status updates for a given exchange (beta)"
    id: pass the exchange id (can be obtained from /exchanges/list) eg. binance
        per_page: Total results per page
        page: Page through results
        
    """
    url = f"https://coingecko.p.rapidapi.com/exchanges/{is_id}/status_updates"
    querystring = {}
    if per_page:
        querystring['per_page'] = per_page
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_list(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all supported coins id, name and symbol (no pagination required)"
    
    """
    url = f"https://coingecko.p.rapidapi.com/coins/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def simple_supported_vs_currencies(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get list of supported_vs_currencies."
    
    """
    url = f"https://coingecko.p.rapidapi.com/simple/supported_vs_currencies"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchanges_id_tickers(is_id: str, page: str=None, coin_ids: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get exchange tickers (paginated)"
    id: pass the exchange id (can be obtained from /exchanges/list) eg. binance
        page: Page through results
        coin_ids: filter tickers by coin_ids (ref: v3/coins/list)
        
    """
    url = f"https://coingecko.p.rapidapi.com/exchanges/{is_id}/tickers"
    querystring = {}
    if page:
        querystring['page'] = page
    if coin_ids:
        querystring['coin_ids'] = coin_ids
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def status_updates(project_type: str=None, category: str=None, page: str=None, per_page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " List all status_updates with data (description, category, created_at, user, user_title and pin)"
    project_type: Filtered by Project Type (eg. coin, market). If left empty returns both status from coins and markets.
        category: Filtered by category (eg. general, milestone, partnership, exchange_listing, software_release, fund_movement, new_listings, event)
        page: Page through results
        per_page: Total results per page
        
    """
    url = f"https://coingecko.p.rapidapi.com/status_updates"
    querystring = {}
    if project_type:
        querystring['project_type'] = project_type
    if category:
        querystring['category'] = category
    if page:
        querystring['page'] = page
    if per_page:
        querystring['per_page'] = per_page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def coins_id_history(date: str, is_id: str, localization: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get historical data (name, price, market, stats) at a given date for a coin"
    date: The date of data snapshot in dd-mm-yyyy eg. 30-12-2017
        id: pass the coin id (can be obtained from /coins) eg. bitcoin
        localization: Set to false to exclude localized languages in response
        
    """
    url = f"https://coingecko.p.rapidapi.com/coins/{is_id}/history"
    querystring = {'date': date, }
    if localization:
        querystring['localization'] = localization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def simple_token_price_id(is_id: str, vs_currencies: str, contract_addresses: str, include_24hr_vol: str='[False]', include_market_cap: str='[False]', include_last_updated_at: str='[False]', include_24hr_change: str='[False]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get current price of tokens (using contract addresses) for a given platform in any other currency that you need."
    id: The id of the platform issuing tokens (Only ethereum is supported for now)
        vs_currencies: vs_currency of coins, comma-separated if querying more than 1 vs_currency *refers to simple/supported_vs_currencies
        contract_addresses: The contract address of tokens, comma separated
        include_24hr_vol: true/false to include 24hr_vol
        include_market_cap: true/false to include market_cap
        include_last_updated_at: true/false to include last_updated_at of price
        include_24hr_change: true/false to include 24hr_change
        
    """
    url = f"https://coingecko.p.rapidapi.com/simple/token_price/{is_id}"
    querystring = {'vs_currencies': vs_currencies, 'contract_addresses': contract_addresses, }
    if include_24hr_vol:
        querystring['include_24hr_vol'] = include_24hr_vol
    if include_market_cap:
        querystring['include_market_cap'] = include_market_cap
    if include_last_updated_at:
        querystring['include_last_updated_at'] = include_last_updated_at
    if include_24hr_change:
        querystring['include_24hr_change'] = include_24hr_change
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coingecko.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

