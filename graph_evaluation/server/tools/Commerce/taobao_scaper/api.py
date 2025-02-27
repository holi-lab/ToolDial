import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getshopitem(pagenum: int, shopid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all items of one special seller. 18 per page."
    shopid: You can get shopId on item page source code. Try ctrl+F. Keyword is shopId
**NOT Seller Id. **
        
    """
    url = f"https://taobao-scaper.p.rapidapi.com/itemSearchShop.php"
    querystring = {'pageNum': pagenum, 'shopId': shopid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-scaper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def searchitem(pagenum: int, q: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search items by keyword, 20 items per page."
    pagenum: Page number of the query. By default is 1.
        q: The keyword of search, for example \\\\\\\"man cloth\\\\\\\" \\\\\\\"男装\\\\\\\"
        
    """
    url = f"https://taobao-scaper.p.rapidapi.com/itemSearch.php"
    querystring = {'pageNum': pagenum, 'q': q, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-scaper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getitem(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a special Taobao Item Json format data via API, including all sub-items."
    id: Taobao or Tmall item id. 
        
    """
    url = f"https://taobao-scaper.p.rapidapi.com/itemGet.php"
    querystring = {'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-scaper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getitemreview(pagenum: int, is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get item reviews. 20 per page."
    id: Taobao or Tmall item id
        
    """
    url = f"https://taobao-scaper.p.rapidapi.com/itemReview.php"
    querystring = {'pageNum': pagenum, 'id': is_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "taobao-scaper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

