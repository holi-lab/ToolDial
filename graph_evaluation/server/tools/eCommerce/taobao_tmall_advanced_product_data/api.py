import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_item_detail_by_item_id(item_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get almost all information about an item by item_id."
    
    """
    url = f"https://taobao-tmall-advanced-product-data.p.rapidapi.com/taobao/advanced/item_detail"
    querystring = {'item_id': item_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-advanced-product-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_real_time_sales(item_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get item real-time sales by 'item_id'"
    
    """
    url = f"https://taobao-tmall-advanced-product-data.p.rapidapi.com/taobao/item/sales/v2"
    querystring = {'item_id': item_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-advanced-product-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_all_items_in_a_shop(page: int, shop_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all items in a shop by 'shop_id'"
    
    """
    url = f"https://taobao-tmall-advanced-product-data.p.rapidapi.com/taobao/shop/items"
    querystring = {'page': page, 'shop_id': shop_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-advanced-product-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_rate_list(item_id: int, page: int=1, rate_type: str=None, page_size: int=10, sort_type: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get item rate list by 'item_id'"
    item_id: item id
        rate_type: -1:Negative Ratings
0:Neutral Ratings
1:Good Ratings
2: There is additional rateing
3: Rating with pictures
**default: all ratings**

        sort_type: feedbackdate:Sort by latest review time
        
    """
    url = f"https://taobao-tmall-advanced-product-data.p.rapidapi.com/taobao/item/rate_list"
    querystring = {'item_id': item_id, }
    if page:
        querystring['page'] = page
    if rate_type:
        querystring['rate_type'] = rate_type
    if page_size:
        querystring['page_size'] = page_size
    if sort_type:
        querystring['sort_type'] = sort_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-advanced-product-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_description_images_and_html(item_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "get item description images and html by 'item_id'"
    
    """
    url = f"https://taobao-tmall-advanced-product-data.p.rapidapi.com/taobao/item_desc"
    querystring = {'item_id': item_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-advanced-product-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

