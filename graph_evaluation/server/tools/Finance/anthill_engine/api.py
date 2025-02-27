import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_klines(market: str, endtime: str=None, limit: int=100, starttime: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint recover klines, a useful tool to generate charts and do analyses on the exchange"
    market: The market to be queried
        endtime: The start time of the search
        limit: The start time of the search
        starttime: The start time of the search
        
    """
    url = f"https://anthill-engine.p.rapidapi.com/analytics"
    querystring = {'market': market, }
    if endtime:
        querystring['endTime'] = endtime
    if limit:
        querystring['limit'] = limit
    if starttime:
        querystring['startTime'] = starttime
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anthill-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_trades(createdat: str=None, pagesize: int=None, page: int=None, market: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get trades history for a tenant market."
    createdat: The oldest date-time to be filter
        pagesize: Page size to be query
        page: Page to be query
        market: The market pair to be query
        
    """
    url = f"https://anthill-engine.p.rapidapi.com/trades"
    querystring = {}
    if createdat:
        querystring['createdAt'] = createdat
    if pagesize:
        querystring['pageSize'] = pagesize
    if page:
        querystring['page'] = page
    if market:
        querystring['market'] = market
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anthill-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_orders(market: str=None, side: str=None, status: str='[]', ordertype: str=None, externalid: int=None, page: int=None, pagesize: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all orders on tenant engine"
    market: The market pair to be query
        side: Order side to be included
        status: Status to be included, by default get all status
        ordertype: The order type to filter
        externalid: External ID to be filter
        page: Page to be query
        pagesize: Page size to be query
        
    """
    url = f"https://anthill-engine.p.rapidapi.com/orders"
    querystring = {}
    if market:
        querystring['market'] = market
    if side:
        querystring['side'] = side
    if status:
        querystring['status'] = status
    if ordertype:
        querystring['orderType'] = ordertype
    if externalid:
        querystring['externalId'] = externalid
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anthill-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_book(market: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get book for a specific tenant market."
    
    """
    url = f"https://anthill-engine.p.rapidapi.com/orders/books/{market}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "anthill-engine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

