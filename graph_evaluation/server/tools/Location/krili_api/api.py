import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_list_of_houses(content_type: str='application/json', bathsmax: int=None, roomsmin: int=None, climat: str=None, wifi: str=None, pricemax: str=None, bathsmin: int=None, pricemin: str=None, rent_frequency: str=None, purpose: str='a-vendre', areamax: int=None, areamin: int=None, roomsmax: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "this endpoint will return back all houses for sale and for rent,"
    
    """
    url = f"https://krili-api.p.rapidapi.com/krili"
    querystring = {}
    if content_type:
        querystring['Content-Type'] = content_type
    if bathsmax:
        querystring['bathsMax'] = bathsmax
    if roomsmin:
        querystring['roomsMin'] = roomsmin
    if climat:
        querystring['climat'] = climat
    if wifi:
        querystring['wifi'] = wifi
    if pricemax:
        querystring['priceMax'] = pricemax
    if bathsmin:
        querystring['bathsMin'] = bathsmin
    if pricemin:
        querystring['priceMin'] = pricemin
    if rent_frequency:
        querystring['rent_frequency'] = rent_frequency
    if purpose:
        querystring['purpose'] = purpose
    if areamax:
        querystring['areaMax'] = areamax
    if areamin:
        querystring['areaMin'] = areamin
    if roomsmax:
        querystring['roomsMax'] = roomsmax
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "krili-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

