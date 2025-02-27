import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def attribute_count(store_key: str, api_key: str, type: str='select', store_id: str='1', lang_id: int=1, visible: bool=None, required: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This method provides you with a possibility to count all attributes in the store. Below you will find all parameters that are necessary to establish attributes count as well as request and response examples. Integration code samples together with interactive documentation (Swagger) that are also presented will make API2Cart a clear tool to use."
    store_key: API2Cart store key
        api_key: API2Cart user key
        type: Defines attribute's type
        store_id: Store Id
        lang_id: Language id
        visible: Filter items by visibility status
        required: Defines if the option is required
        
    """
    url = f"https://magneticone-api2cart.p.rapidapi.com/attribute.count.json"
    querystring = {'store_key': store_key, 'api_key': api_key, }
    if type:
        querystring['type'] = type
    if store_id:
        querystring['store_id'] = store_id
    if lang_id:
        querystring['lang_id'] = lang_id
    if visible:
        querystring['visible'] = visible
    if required:
        querystring['required'] = required
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magneticone-api2cart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def cart_giftcard_count(store_key: str, api_key: str, store_id: str='2', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Using this method you can get the number of giftcards on the particular store. Below you will find all parameters that are necessary to establish giftcards' count as well as request and response examples. Integration code samples together with interactive documentation (Swagger) that are also presented will make API2Cart a clear tool to use."
    store_key: API2Cart store key
        api_key: API2Cart user key
        store_id: Store Id
        
    """
    url = f"https://magneticone-api2cart.p.rapidapi.com/cart.giftcard.count.json"
    querystring = {'store_key': store_key, 'api_key': api_key, }
    if store_id:
        querystring['store_id'] = store_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magneticone-api2cart.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

