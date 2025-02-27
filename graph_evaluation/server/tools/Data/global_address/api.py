import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def global_address(ctry: str, format: str='json', a5: str=None, a1: str='Gießener Str. 30', deliverylines: str='Off', a2: str='Frankfurt am Main', a3: str=None, admarea: str=None, a7: str=None, a6: str=None, postal: str='60435', a8: str=None, loc: str=None, a4: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Validate, Standardize and Cleanse Addresses across the globe"
    ctry: Country Code / Country Name
        format: Format of the Response (XML or JSON)
        a5: Address Line 5
        a1: Address Line 1
        deliverylines: Specify if the Address Lines 1-8 should contain just the delivery address or entire address.
        a2: Address Line 2
        a3: Address Line 3
        admarea: Administrative Area
        a7: Address Line 7
        a6: Address Line 6
        postal: Postal Code
        a8: Address Line 8
        loc: Locality
        a4: Address Line 4
        
    """
    url = f"https://global-address.p.rapidapi.com/V3/WEB/GlobalAddress/doGlobalAddress"
    querystring = {'ctry': ctry, }
    if format:
        querystring['format'] = format
    if a5:
        querystring['a5'] = a5
    if a1:
        querystring['a1'] = a1
    if deliverylines:
        querystring['DeliveryLines'] = deliverylines
    if a2:
        querystring['a2'] = a2
    if a3:
        querystring['a3'] = a3
    if admarea:
        querystring['admarea'] = admarea
    if a7:
        querystring['a7'] = a7
    if a6:
        querystring['a6'] = a6
    if postal:
        querystring['postal'] = postal
    if a8:
        querystring['a8'] = a8
    if loc:
        querystring['loc'] = loc
    if a4:
        querystring['a4'] = a4
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "global-address.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

