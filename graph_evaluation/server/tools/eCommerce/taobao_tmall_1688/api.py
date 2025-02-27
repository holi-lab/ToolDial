import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_item_detail(num_id: int, provider: str, is_promotion: str='1', lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the item detail using item id."
    num_id: A unique item id
        provider: 1688 or taobao
        is_promotion: Use promotion price? 
1 = yes
0 = no
        lang: valid value en, cn, ru
        
    """
    url = f"https://taobao-tmall-16882.p.rapidapi.com/item_get"
    querystring = {'num_id': num_id, 'provider': provider, }
    if is_promotion:
        querystring['is_promotion'] = is_promotion
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-16882.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_item(provider: str, end_price: str, start_price: str, keyword: str, cat: str='0', sort: str='_sale', page: str='20', lang: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this end point to search item on taobao, 1688, TMALL"
    cat: Cat is category ID
        sort: Sort by
available value
bid,_bid,bid2,_bid2,_sale,_credit
        page: Size of response
Maximum is 20
        lang: Select response language
available value
en,cn,ru
        
    """
    url = f"https://taobao-tmall-16882.p.rapidapi.com/item_search"
    querystring = {'provider': provider, 'end_price': end_price, 'start_price': start_price, 'keyword': keyword, }
    if cat:
        querystring['cat'] = cat
    if sort:
        querystring['sort'] = sort
    if page:
        querystring['page'] = page
    if lang:
        querystring['lang'] = lang
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-tmall-16882.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

