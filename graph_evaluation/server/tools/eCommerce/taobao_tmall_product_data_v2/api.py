import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_item_detail_by_item_id(item_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get pruduct details by 'item_id';"
    
    """
    url = f"https://taobao-tmall-product-data-v2.p.rapidapi.com/api/sc/taobao/item_detail"
    querystring = {'item_id': item_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-product-data-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_description_information(item_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get item description images and html by item_id"
    
    """
    url = f"https://taobao-tmall-product-data-v2.p.rapidapi.com/api/sc/taobao/item_desc"
    querystring = {'item_id': item_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-product-data-v2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

