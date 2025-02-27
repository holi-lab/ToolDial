import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def short(url: str, key: str, address: str='Deublerstreet 37', lng: str='16.3909', lat: str='48.2675', note: str='2nd floor, left door', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The method shortens a given URL and returns the shortended URL as well the QR Code URL. You may attach optional geo information to set a location of where the QR Code will be placed. This may be your company address or any location where you want to stick your QR Code. The parameter address is a human readable address string. The parameter lat (Latitude) and lng (Longitude) are used for further machine processing, e.g. to show the location on a map. The note parameter is used to add a detailed description about the location (e.g. 2nd floor, left door)."
    url: The URL to shorten
        key: The API Key
        address: An optional human readable address string where the QR Code will be attached
        lng: An optional longitude of where  the QR Code will be attached
        lat: An optional latitude of where  the QR Code will be attached
        note: An optional note
        
    """
    url = f"https://qrd.p.rapidapi.com/short"
    querystring = {'url': url, 'key': key, }
    if address:
        querystring['address'] = address
    if lng:
        querystring['lng'] = lng
    if lat:
        querystring['lat'] = lat
    if note:
        querystring['note'] = note
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "qrd.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

