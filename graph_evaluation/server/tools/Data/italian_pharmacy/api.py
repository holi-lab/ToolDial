import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getpharmaciesusingget(size: int=20, filteror: str=None, orders: str=None, filterand: str=None, page: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "getPharmaciesUsingGET"
    size: size
        filteror: filterOr
        orders: orders
        filterand: filterAnd
        page: page
        
    """
    url = f"https://italian-pharmacy.p.rapidapi.com/pharmacy"
    querystring = {}
    if size:
        querystring['size'] = size
    if filteror:
        querystring['filterOr'] = filteror
    if orders:
        querystring['orders'] = orders
    if filterand:
        querystring['filterAnd'] = filterand
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "italian-pharmacy.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

