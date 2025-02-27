import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_global_stats(referencecurrencyuuid: str='yhjMzLPhuIDl', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "These global statistics tell about the data available on coinranking."
    referencecurrencyuuid: UUID of reference currency, in which all the prices are calculated. Defaults to US Dollar

Default value: yhjMzLPhuIDl
        
    """
    url = f"https://coinranking1.p.rapidapi.com/stats"
    querystring = {}
    if referencecurrencyuuid:
        querystring['referenceCurrencyUuid'] = referencecurrencyuuid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_exchanges_index(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all exchanges currently available on Coinranking, for indexing purposes.
		This endpoint requires the **ultra** plan or higher."
    
    """
    url = f"https://coinranking1.p.rapidapi.com/indexes/exchanges"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_coins_index(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List of all coins currently available on coinranking, for indexing purposes.
		This endpoint requires the **ultra** plan or higher."
    
    """
    url = f"https://coinranking1.p.rapidapi.com/indexes/coins"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_market(uuid: str, referencecurrencyuuid: str='yhjMzLPhuIDl', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find information on a specific market listed on Coinranking.
		This endpoint requires the **ultra** plan or higher."
    uuid: Uuid of the market you want to request
        referencecurrencyuuid: Uuid of reference currency, in which all the prices are calculated. Defaults to US Dollar

Default value: yhjMzLPhuIDl
        
    """
    url = f"https://coinranking1.p.rapidapi.com/market/{uuid}"
    querystring = {}
    if referencecurrencyuuid:
        querystring['referenceCurrencyUuid'] = referencecurrencyuuid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_exchange_coins(uuid: str, offset: int=0, search: str=None, orderby: str='24hVolume', orderdirection: str='desc', referencecurrencyuuid: str='yhjMzLPhuIDl', limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find coins listed on a specific exchange.
		This endpoint requires the **ultra** plan or higher."
    uuid: UUID of the exchange you want to request
        offset: Offset. Used for pagination

Default value: 0
        search: Filter the results by searching for coin names or symbols.
        orderby: Index to sort on. Default is 24h volume

Default value: 24hVolume
Allowed values:
24hVolume price numberOfMarkets
        orderdirection: Order in ascending or descending order

Default value: desc
Allowed values:
asc desc
        referencecurrencyuuid: UUID of reference currency, which rate is used to calculate the volume. Defaults to US Dollar

Default value: yhjMzLPhuIDl
        limit: Limit. Used for pagination

Default value: 50
Size range: 0-100
        
    """
    url = f"https://coinranking1.p.rapidapi.com/exchange/{uuid}/coins"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if search:
        querystring['search'] = search
    if orderby:
        querystring['orderBy'] = orderby
    if orderdirection:
        querystring['orderDirection'] = orderdirection
    if referencecurrencyuuid:
        querystring['referenceCurrencyUuid'] = referencecurrencyuuid
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_exchange(uuid: str, referencecurrencyuuid: str='yhjMzLPhuIDl', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find information on a specific exchange listed on coinranking. An exchange is a place where cryptocurrencies are traded.
		This endpoint requires the **ultra** plan or higher."
    uuid: UUID of the exchange you want to request
        referencecurrencyuuid: UUID of reference currency, which rate is used to calculate the volume. Defaults to US Dollar

Default value: yhjMzLPhuIDl
        
    """
    url = f"https://coinranking1.p.rapidapi.com/exchange/{uuid}"
    querystring = {}
    if referencecurrencyuuid:
        querystring['referenceCurrencyUuid'] = referencecurrencyuuid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_coin_modifiers(uuid: str, offset: int=0, limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the modifiers of a coin's supply and their balance.
		This endpoint requires the **ultra** plan or higher."
    uuid: UUID of the coin you want to request the modifiers for
        offset: Offset. Used for pagination

Default value: 0
        limit: Limit. Used for pagination

Default value: 50
Size range: 0-100
        
    """
    url = f"https://coinranking1.p.rapidapi.com/coin/{uuid}/modifiers"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_coin_exchanges(uuid: str, search: str=None, orderby: str='24hVolume', orderdirection: str='desc', offset: int=0, limit: int=50, referencecurrencyuuid: str='yhjMzLPhuIDl', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find exchanges where a specific coin can be traded.
		This endpoint requires the **ultra** plan or higher."
    uuid: UUID of the coin you want to request exchanges for
        search: Value to search for within results, i.e. exchange names
        orderby: Index to order by. Default is 24h volume.

Default value: 24hVolume
Allowed values:
24hVolume price
        orderdirection: Order in ascending or descending order

Default value: desc
Allowed values:
desc asc
        offset: Offset. Used for pagination

Default value: 0
        limit: Limit. Used for pagination

Default value: 50
Size range: 0-100
        referencecurrencyuuid: UUID of reference currency, which rate is used to calculate the volume. Defaults to US Dollar

Default value: yhjMzLPhuIDl
        
    """
    url = f"https://coinranking1.p.rapidapi.com/coin/{uuid}/exchanges"
    querystring = {}
    if search:
        querystring['search'] = search
    if orderby:
        querystring['orderBy'] = orderby
    if orderdirection:
        querystring['orderDirection'] = orderdirection
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    if referencecurrencyuuid:
        querystring['referenceCurrencyUuid'] = referencecurrencyuuid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_coin_ohlc_data(uuid: str, referencecurrencyuuid: str='yhjMzLPhuIDl', limit: int=None, interval: str='day', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get OHLC (Open High Low Close) data for the coin throughout time.
		This endpoint requires the **ultra** plan or higher.
		
		**Beta**
		The OHLC endpoint is currently in beta. This means we might make some changes that could be considered breaking for your application, and we expect to have downtime every now and then while we are still in beta."
    uuid: UUID of the coin you want to request. UUIDs of coins can be found using the Get coins endpoint or by checking the URL on coinranking.com, e.g. https://coinranking.com/coin/Qwsogvtv82FCd+bitcoin-btc is the URL for Bitcoin, and the part before the + (Qwsogvtv82FCd) is the UUID.
        referencecurrencyuuid: UUID of reference currency, in which all the prices are calculated. Defaults to US Dollar.

Default value: yhjMzLPhuIDl
        limit: Limit the amount of time periods for which the OHLC data is retrieved. For example, when interval=hour and limit is 10, data will be returned for the last 10 hours.

Default value: 50
Size range: 0-100
        interval: The interval determines the time period over which each OHLC item is determined.

Default value: day
Allowed values:
minute 5minutes hour 8hours day week month
        
    """
    url = f"https://coinranking1.p.rapidapi.com/coin/{uuid}/ohlc"
    querystring = {}
    if referencecurrencyuuid:
        querystring['referenceCurrencyUuid'] = referencecurrencyuuid
    if limit:
        querystring['limit'] = limit
    if interval:
        querystring['interval'] = interval
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_coin(uuid: str, timeperiod: str='24h', referencecurrencyuuid: str='yhjMzLPhuIDl', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find information about a specific coin."
    uuid: UUID of the coin you want to request. UUIDs of coins can be found using the Get coins endpoint or by checking the URL on coinranking.com, e.g. https://coinranking.com/coin/Qwsogvtv82FCd+bitcoin-btc is the URL for Bitcoin, and the part before the + (Qwsogvtv82FCd) is the UUID.
        timeperiod: Time period where the change and sparkline are based on

Default value: 24h
Allowed values:
24h 7d 30d
        referencecurrencyuuid: UUID of reference currency, in which all the prices are calculated. Defaults to US Dollar

Default value: yhjMzLPhuIDl
        
    """
    url = f"https://coinranking1.p.rapidapi.com/coin/{uuid}"
    querystring = {}
    if timeperiod:
        querystring['timePeriod'] = timeperiod
    if referencecurrencyuuid:
        querystring['referenceCurrencyUuid'] = referencecurrencyuuid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_coin_price(uuid: str, timestamp: int=None, referencecurrencyuuid: str='yhjMzLPhuIDl', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With the price endpoint the price can be requested for a specific coin on a specific time. The response is just a single price nearest to the requested time, including its timestamp."
    uuid: UUID of the coin you need the price
        timestamp: Timestamp. Epoch timestamp in seconds. If it is not provided this endpoint will get the latest price
        referencecurrencyuuid: UUID of reference currency. This is the currency the price is shown in, which defaults to US Dollar

Default value: yhjMzLPhuIDl
        
    """
    url = f"https://coinranking1.p.rapidapi.com/coin/{uuid}/price"
    querystring = {}
    if timestamp:
        querystring['timestamp'] = timestamp
    if referencecurrencyuuid:
        querystring['referenceCurrencyUuid'] = referencecurrencyuuid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_coin_price_history(uuid: str, referencecurrencyuuid: str='yhjMzLPhuIDl', timeperiod: str='24h', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Coinranking keeps track of prices on all listed assets. The history endpoint lists prices and their timestamp for the requested time period, useful for making a chart."
    uuid: UUID of the coin you want to request
        referencecurrencyuuid: UUID of reference currency, in which all the prices are calculated. Defaults to US Dollar

Default value: yhjMzLPhuIDl
        timeperiod: Timeperiod where the change and history are based on

Default value: 24h
Allowed values:
3h 24h 7d 30d 3m 1y 3y 5y
        
    """
    url = f"https://coinranking1.p.rapidapi.com/coin/{uuid}/history"
    querystring = {}
    if referencecurrencyuuid:
        querystring['referenceCurrencyUuid'] = referencecurrencyuuid
    if timeperiod:
        querystring['timePeriod'] = timeperiod
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_search_suggestions(query: str=None, referencecurrencyuuid: str='yhjMzLPhuIDl', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search suggestions are a quick and easy way to find data on coinranking. The endpoint only accepts one parameter; a query. With this query you can find currencies (including fiat), exchanges and markets, by their symbol or name. The response always returns a set of the most prominent coins, exchanges and markets matching your query."
    query: Value to search on
        referencecurrencyuuid: UUID of reference currency, in which the coin prices are calculated. Defaults to US Dollar

Default value: yhjMzLPhuIDl
        
    """
    url = f"https://coinranking1.p.rapidapi.com/search-suggestions"
    querystring = {}
    if query:
        querystring['query'] = query
    if referencecurrencyuuid:
        querystring['referenceCurrencyUuid'] = referencecurrencyuuid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_markets(tocurrencyuuid: str=None, quotecurrencyuuid: str=None, limit: int=50, offset: int=0, referencecurrencyuuid: str='yhjMzLPhuIDl', basecurrencyuuid: str=None, orderdirection: str='desc', orderby: str='24hVolume', search: str=None, currencyuuid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of markets. Markets are ranked by their volume over the last 24 hours. Use our filters to get a subset of the markets.
		This endpoint requires the **ultra** plan or higher."
    tocurrencyuuid: Filter markets with specific currency as either base or quote. The toCurrencyUuid will not alter how the prices will be shown, but will keep the base price. This can be combined with the currencyUuid variable to get specific markets.
        quotecurrencyuuid: Filter markets with specific currency as quote
        limit: Limit. Used for pagination. Only usable when no filters are applied

Default value: 50
Size range: 0-100
        offset: Offset. Used for pagination only usable when no filters are applied

Default value: 0
        referencecurrencyuuid: UUID of reference currency, in which all the prices are calculated. Defaults to US Dollar

Default value: yhjMzLPhuIDl
        basecurrencyuuid: Filter markets with specific currency as base
        orderdirection: Sort in ascending or descending order. Only usable when no filters are applied.

Default value: desc
Allowed values:
desc asc
        orderby: Sort by either 24h volume or price. Only usable when no filters are applied

Default value: 24hVolume
Allowed values:
24hVolume price
        search: Filter the results by searching for coin names, symbols or exchange names.
        currencyuuid: Filter markets with specific currency as either base or quote. Specifying a currencyUuid will also alter how prices are shown: By default all the markets will show the price of the base in the reference currency (e.g. an ETH/BTC market will show the price of ETH). By specifying a currencyUuid the prices of this currency will always be shown, disregarding whether or not this currency represents the base or the quote in the market (e.g. by specifying BTC as currency, both ETH/BTC as BTC/USD markets will show prices of BTC)
        
    """
    url = f"https://coinranking1.p.rapidapi.com/markets"
    querystring = {}
    if tocurrencyuuid:
        querystring['toCurrencyUuid'] = tocurrencyuuid
    if quotecurrencyuuid:
        querystring['quoteCurrencyUuid'] = quotecurrencyuuid
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if referencecurrencyuuid:
        querystring['referenceCurrencyUuid'] = referencecurrencyuuid
    if basecurrencyuuid:
        querystring['baseCurrencyUuid'] = basecurrencyuuid
    if orderdirection:
        querystring['orderDirection'] = orderdirection
    if orderby:
        querystring['orderBy'] = orderby
    if search:
        querystring['search'] = search
    if currencyuuid:
        querystring['currencyUuid'] = currencyuuid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_exchange_markets(uuid: str, referencecurrencyuuid: str='yhjMzLPhuIDl', offset: int=0, orderby: str='24hVolume', search: str=None, limit: int=50, orderdirection: str='desc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find markets on a specific exchange.
		This endpoint requires the **ultra** plan or higher."
    uuid: UUID of the exchange you want to request markets for
        referencecurrencyuuid: UUID of reference currency, which rate is used to calculate the volume. Defaults to US Dollar

Default value: yhjMzLPhuIDl
        offset: Offset. Used for pagination

Default value: 0
        orderby: Index to sort on. Default is 24h volume.

Default value: 24hVolume
Allowed values:
24hVolume price
        search: Value to search for within results, e.g. exchange names, currency names, or currency symbols
        limit: Limit. Used for pagination

Default value: 50
Size range: 0-100
        orderdirection: Order in ascending or descending order

Default value: desc
Allowed values:
desc asc
        
    """
    url = f"https://coinranking1.p.rapidapi.com/exchange/{uuid}/markets"
    querystring = {}
    if referencecurrencyuuid:
        querystring['referenceCurrencyUuid'] = referencecurrencyuuid
    if offset:
        querystring['offset'] = offset
    if orderby:
        querystring['orderBy'] = orderby
    if search:
        querystring['search'] = search
    if limit:
        querystring['limit'] = limit
    if orderdirection:
        querystring['orderDirection'] = orderdirection
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_coin_issuance_blockchains(uuid: str, offset: int=0, limit: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the issuance blockchains on which the coin is issued.
		This endpoint requires the **ultra** plan or higher."
    uuid: UUID of the coin you want to request the blockchains for
        offset: Offset. Used for pagination

Default value: 0
        limit: Limit. Used for pagination

Default value: 50
Size range: 0-100
        
    """
    url = f"https://coinranking1.p.rapidapi.com/coin/{uuid}/issuance-blockchains"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_coin_supply(uuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the maximum, total, and circulating supply of a coin."
    uuid: UUID of the coin you want to request the supply for
        
    """
    url = f"https://coinranking1.p.rapidapi.com/coin/{uuid}/supply"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_coin_markets(uuid: str, offset: int=0, orderby: str='24hVolume', limit: int=50, search: str=None, referencecurrencyuuid: str='yhjMzLPhuIDl', orderdirection: str='desc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find markets on different exchanges that trade a specific coin.
		This endpoint requires the **ultra** plan or higher."
    uuid: UUID of the coin you want to request markets for
        offset: Offset. Used for pagination

Default value: 0
        orderby: Index to sort on. Default is 24h volume.

Default value: 24hVolume
Allowed values:
24hVolume price
        limit: Limit. Used for pagination

Default value: 50
Size range: 0-100
        search: Value to search for within results, e.g. exchange names, currency names, or currency symbols
        referencecurrencyuuid: UUID of reference currency, which rate is used to calculate the volume. Defaults to US Dollar

Default value: yhjMzLPhuIDl
        orderdirection: Order in ascending or descending order

Default value: desc
Allowed values:
desc asc
        
    """
    url = f"https://coinranking1.p.rapidapi.com/coin/{uuid}/markets"
    querystring = {}
    if offset:
        querystring['offset'] = offset
    if orderby:
        querystring['orderBy'] = orderby
    if limit:
        querystring['limit'] = limit
    if search:
        querystring['search'] = search
    if referencecurrencyuuid:
        querystring['referenceCurrencyUuid'] = referencecurrencyuuid
    if orderdirection:
        querystring['orderDirection'] = orderdirection
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "coinranking1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

