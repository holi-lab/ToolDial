import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def taobao_item_search(q: str, sort: str=None, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "New Version Item Search"
    
    """
    url = f"https://taobao-advanced.p.rapidapi.com/item_search_3"
    querystring = {'q': q, }
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-advanced.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shop_items(seller_id: int, api: str, q: str=None, page: int=1, sort: str='default', cat: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API gets you all the items within the Seller's Shop. Combining with the shop's categories API, you can collect and represent the best and comfortable feeling of shopping and sorting items."
    seller_id: Seller Id
        api: Do Not Edit This Field
        sort: Returned results sorting. Available options: Best match (default), Sales descending (sales_des), Price ascending (price_asc), Price descending (price_des), New arrivals first (new_arrivals).
        
    """
    url = f"https://taobao-advanced.p.rapidapi.com/api"
    querystring = {'seller_id': seller_id, 'api': api, }
    if q:
        querystring['q'] = q
    if page:
        querystring['page'] = page
    if sort:
        querystring['sort'] = sort
    if cat:
        querystring['cat'] = cat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-advanced.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_detail_new_with_promo_price(api: str, num_iid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Advanced Item Detail API that includes all the information about the product, information about the seller, shipping prices, SKU's description images, videos, etc."
    api: Don’t edit this field
        num_iid: Taobao/Tmall Item ID.
        
    """
    url = f"https://taobao-advanced.p.rapidapi.com/api"
    querystring = {'api': api, 'num_iid': num_iid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-advanced.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_review(api: str, num_iid: int, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Along with the Advanced Item Detail API, this API shows all the reviews for this selected item completing the whole package of information about the product possible."
    api: Don’t edit this field
        num_iid: Taobao/Tmall Item ID.
        
    """
    url = f"https://taobao-advanced.p.rapidapi.com/api"
    querystring = {'api': api, 'num_iid': num_iid, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-advanced.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_detail(num_iid: int, api: str, area_id: int=110100, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Advanced Item Detail API that includes all the information about the product, information about the seller, shipping prices, SKU's description images, videos, etc."
    num_iid: Taobao/Tmall Item ID.
        api: Don’t edit this field
        
    """
    url = f"https://taobao-advanced.p.rapidapi.com/api"
    querystring = {'num_iid': num_iid, 'api': api, }
    if area_id:
        querystring['area_id'] = area_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-advanced.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_search_by_image(img: str, api: str, region: str=None, cat: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this API you will be able to get all related items to the image provided. Keep in must image must be from Taobao to work."
    img: Any image url that is from Taobao.
        api: Do NOT Edit this field.
        region: The specific location from an image that search is focused on.
        cat: Sorting, from response.
        
    """
    url = f"https://taobao-advanced.p.rapidapi.com/api"
    querystring = {'img': img, 'api': api, }
    if region:
        querystring['region'] = region
    if cat:
        querystring['cat'] = cat
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-advanced.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_recommended(guang_id: int, api: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shopping Lovers Item Recommended"
    guang_id: Obtained from the Shopping Lovers Item List Endpoint
        api: Do Not Edit This Field.
        page: Page number, starts from page 1. Default page is 1.
        
    """
    url = f"https://taobao-advanced.p.rapidapi.com/api"
    querystring = {'guang_id': guang_id, 'api': api, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-advanced.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_list(api: str, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns some nice looking products."
    api: Don’t edit this field
        page: Page number, starts from page 1. Default page is 1
        
    """
    url = f"https://taobao-advanced.p.rapidapi.com/api"
    querystring = {'api': api, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-advanced.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def item_details(api: str, guang_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Shopping lovers Item details"
    guang_id: Obtained from the Shopping Lovers Item List Endpoint
        
    """
    url = f"https://taobao-advanced.p.rapidapi.com/api"
    querystring = {'api': api, 'guang_id': guang_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-advanced.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def shop_categories(api: str, seller_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets the list of categories within the seller's shop. This API should be combined with Shop Items API to pass the details of categories for items filtering."
    
    """
    url = f"https://taobao-advanced.p.rapidapi.com/api"
    querystring = {'api': api, 'seller_id': seller_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-advanced.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

