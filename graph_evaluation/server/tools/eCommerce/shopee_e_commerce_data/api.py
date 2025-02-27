import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_shop_items(shop_id: int, site: str, by: str=None, keyword: str=None, order: str=None, pagesize: int=20, page: int=1, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all items of a Shopee shop bshop id"
    by: This takes effect only when the **keyword **is not empty
        order: This takes effect only when **by=price**
        
    """
    url = f"https://shopee-e-commerce-data.p.rapidapi.com/shopee/shop/items/v2"
    querystring = {'shop_id': shop_id, 'site': site, }
    if by:
        querystring['by'] = by
    if keyword:
        querystring['keyword'] = keyword
    if order:
        querystring['order'] = order
    if pagesize:
        querystring['pageSize'] = pagesize
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopee-e-commerce-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_shops_by_keyword(site: str, keyword: str, page: int=1, pagesize: int=20, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "search shopee shops by keyword"
    
    """
    url = f"https://shopee-e-commerce-data.p.rapidapi.com/shopee/search/shops"
    querystring = {'site': site, 'keyword': keyword, }
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pageSize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopee-e-commerce-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_items_by_keyword(site: str, keyword: str, order: str=None, pagesize: int=60, page: int=1, by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET search items by keyword v2"
    site: Optional values: my, th, vn, ph, sg, id, tw, br
        
    """
    url = f"https://shopee-e-commerce-data.p.rapidapi.com/shopee/search/items/v2"
    querystring = {'site': site, 'keyword': keyword, }
    if order:
        querystring['order'] = order
    if pagesize:
        querystring['pageSize'] = pagesize
    if page:
        querystring['page'] = page
    if by:
        querystring['by'] = by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopee-e-commerce-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_search_hints(keyword: str, site: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET search hints by keyword"
    site: Optional values: my, th, vn, ph, sg, id, tw, br
        
    """
    url = f"https://shopee-e-commerce-data.p.rapidapi.com/shopee/search/hints"
    querystring = {'keyword': keyword, 'site': site, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopee-e-commerce-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_product_detail_by_id_v2(item_id: int, site: str, shop_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET product detail by 'item_id' and 'shop_id' v2"
    item_id: if the product url is [https://shopee.com.my/Xiaomi-Redmi-AirDots-TWS-Mi-True-Wireless-EarBuds-Basic-Earphone-Bluetooth-5.0-Bass-Voice-Control-(BLACK)-i.70413398.7041129024?ads_keyword=wkdaelpmissisiht&adsid=10115290&campaignid=5587639&position=120](url),then the item_id is 7041129024,shop_id is 70413398
        site: Optional values: my, th, vn, ph, sg, id, tw, br
        shop_id: if the product url is [https://shopee.com.my/Xiaomi-Redmi-AirDots-TWS-Mi-True-Wireless-EarBuds-Basic-Earphone-Bluetooth-5.0-Bass-Voice-Control-(BLACK)-i.70413398.7041129024?ads_keyword=wkdaelpmissisiht&adsid=10115290&campaignid=5587639&position=120](url),then the item_id is 7041129024,shop_id is 70413398
        
    """
    url = f"https://shopee-e-commerce-data.p.rapidapi.com/shopee/item_detail/v2"
    querystring = {'item_id': item_id, 'site': site, 'shop_id': shop_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopee-e-commerce-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_shop_detail(site: str, username: str='fangzhong.my', shop_id: int=768887972, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "GET shop detail  information by shop_id or username"
    site: Optional values: my, th, vn, ph, sg, id, tw, br
        
    """
    url = f"https://shopee-e-commerce-data.p.rapidapi.com/shopee/shop/shop_info"
    querystring = {'site': site, }
    if username:
        querystring['username'] = username
    if shop_id:
        querystring['shop_id'] = shop_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopee-e-commerce-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

