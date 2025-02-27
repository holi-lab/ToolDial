import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_a_shop_s_products_by_seller_id(page: str, seller_id: int, sort: str=None, page_size: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "​Get a shop's products by seller id"
    
    """
    url = f"https://taobao-tmall-product-data1.p.rapidapi.com/taobao/shop/items/v3"
    querystring = {'page': page, 'seller_id': seller_id, }
    if sort:
        querystring['sort'] = sort
    if page_size:
        querystring['page_size'] = page_size
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-product-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_s_descriptive_images_and_html_by_id(item_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get product's descriptive images and html by id"
    
    """
    url = f"https://taobao-tmall-product-data1.p.rapidapi.com/taobao/item_desc"
    querystring = {'item_id': item_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-product-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_details_by_id(item_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get product details by id"
    item_id: e.g.:
if the item url is :https://detail.tmall.com/item.htm?id=670674630083&spm=a21n57.1.0.0,
then item_id=670674630083
        
    """
    url = f"https://taobao-tmall-product-data1.p.rapidapi.com/taobao/item_detail"
    querystring = {'item_id': item_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-product-data1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

