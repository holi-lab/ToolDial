import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def articles(locale: str, size: str='20', page: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Article List"
    locale: Language, support zh_CN (Chinese), en_US (English), ko_KR (Korean)
        size: Per page size, default 20 (>=1, <=20)
        page: Current page, default 0, (>=0)
        
    """
    url = f"https://blockcc1.p.rapidapi.com/api/v3/articles"
    querystring = {'locale': locale, }
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockcc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def brief(locale: str, size: str='20', page: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Briefs"
    locale: Language, support zh_CN (Chinese), en_US (English), ko_KR (Korean)
        size: Per page size, default 20 (>=1)
        page: Current page, default 0, (>=0)
        
    """
    url = f"https://blockcc1.p.rapidapi.com/api/v3/briefs"
    querystring = {'locale': locale, }
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockcc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def price(page: str='0', slug: str='bitcoin,filecoin', size: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get currency price"
    page: Current page, default 0, (>=0)
        slug: Currency slug, comma-separated
        size: Per page size, default is 20 (100>=size>=1)
        
    """
    url = f"https://blockcc1.p.rapidapi.com/api/v3/price"
    querystring = {}
    if page:
        querystring['page'] = page
    if slug:
        querystring['slug'] = slug
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockcc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historicalprice(slug: str, end: str=None, start: str=None, interval: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get currency historical price"
    slug: Symbol Slug.
        end: End Time，unit: mills，default current timestamp
        start: Start Time，unit: mills, default min timestamp
        interval: Interval[5m,15m,30m,1h,2h,6h,12h,1d,2d], default calculated based on start, end
        
    """
    url = f"https://blockcc1.p.rapidapi.com/api/v3/price/history"
    querystring = {'slug': slug, }
    if end:
        querystring['end'] = end
    if start:
        querystring['start'] = start
    if interval:
        querystring['interval'] = interval
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockcc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def announcements(locale: str, page: str='0', size: str='20', market: str='binance', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Exchange Announcement List"
    locale: Language, support zh_CN (Chinese), en_US (English), ko_KR (Korean)
        page: Current page, default 0, (>=0)
        size: Per page size, default 20 (>=1)
        market: Market slug
        
    """
    url = f"https://blockcc1.p.rapidapi.com/api/v3/announcements"
    querystring = {'locale': locale, }
    if page:
        querystring['page'] = page
    if size:
        querystring['size'] = size
    if market:
        querystring['market'] = market
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockcc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbols(page: str='0', details: str='0', size: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all supported currencies"
    page: Current page, default 0, (>=0).
        details: Is getting the details of the currency, the value is 1(yes), 0(no), the default is 0
        size: Per page size, default is 20 (100>=size>=1).
        
    """
    url = f"https://blockcc1.p.rapidapi.com/api/v3/symbols"
    querystring = {}
    if page:
        querystring['page'] = page
    if details:
        querystring['details'] = details
    if size:
        querystring['size'] = size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockcc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def market(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single market"
    slug: Get single market,e.g. /api/v3/markets/gate-io
        
    """
    url = f"https://blockcc1.p.rapidapi.com/api/v3/markets/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockcc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def orderbook(desc: str, limit: str='25', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get OrderBook"
    desc: MarketPair Desc e.g. gate-io_BTC_USDT
        limit: Limit,default 25.
        
    """
    url = f"https://blockcc1.p.rapidapi.com/api/v3/orderbook"
    querystring = {'desc': desc, }
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockcc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def kline_candlestick(desc: str, interval: str='5m', end: str=None, start: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Kline/Candlestick Data(OHLCV)"
    desc: MarketPair Desc e.g. gate-io_BTC_USDT
        interval: interval [1m,5m,15m,30m,1h,6h,1d], default: 5m
        end: End Time，unit: mills，default current time
        start: Start Time，unit: mills，default end - (1000 * interval)
        
    """
    url = f"https://blockcc1.p.rapidapi.com/api/v3/kline"
    querystring = {'desc': desc, }
    if interval:
        querystring['interval'] = interval
    if end:
        querystring['end'] = end
    if start:
        querystring['start'] = start
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockcc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def exchangerate(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the exchange rate, the exchange rate of this interface is based on `USD`"
    
    """
    url = f"https://blockcc1.p.rapidapi.com/api/v3/exchange_rate"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockcc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def symbol(slug: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single symbol"
    slug: for getting single currency
        
    """
    url = f"https://blockcc1.p.rapidapi.com/api/v3/symbols/{slug}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockcc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def markets(size: str='20', page: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of all supported markets"
    size: Per page size, default is 20 (100>=size>=1)
        page: Current page, default 0, (>=0)
        
    """
    url = f"https://blockcc1.p.rapidapi.com/api/v3/markets"
    querystring = {}
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockcc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def article(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get One Article"
    id: Article ID
        
    """
    url = f"https://blockcc1.p.rapidapi.com/api/v3/articles/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockcc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tickers(page: str='0', market: str='bitfinex', size: str='20', market_pair: str=None, slug: str=None, symbol: str=None, currency: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Tickers"
    page: Current page, default 0, (>=0)
        market: Market Slug, comma-separated
        size: Per page size, default 20 (>=1)
        market_pair: MarketPairDesc, comma-separated
        slug: BaseCurrency Slug, comma-separated
        symbol: BaseCurrency Symbol, comma-separated
        currency: QuoteCurrency Symbol, comma-separated
        
    """
    url = f"https://blockcc1.p.rapidapi.com/api/v3/tickers"
    querystring = {}
    if page:
        querystring['page'] = page
    if market:
        querystring['market'] = market
    if size:
        querystring['size'] = size
    if market_pair:
        querystring['market_pair'] = market_pair
    if slug:
        querystring['slug'] = slug
    if symbol:
        querystring['symbol'] = symbol
    if currency:
        querystring['currency'] = currency
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockcc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def socialmedia(source: str, size: str='20', page: str='0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get SocialMedia List"
    source: Source: [WEIBO, TWITTER], default WEIBO
        size: Per page size, default is 20 (100>=size>=1)
        page: Current page, default 0, (>=0)
        
    """
    url = f"https://blockcc1.p.rapidapi.com/api/v3/social_media"
    querystring = {'source': source, }
    if size:
        querystring['size'] = size
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "blockcc1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

