import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def show_item_by_id(itemid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`item` api parameters:
		- itemId: The item id you want to query. The item id is something like `m72639077322`, you can get it from the response of `items` API."
    itemid: The id of the item to retrieve
        
    """
    url = f"https://mercari-item-search.p.rapidapi.com/item/{itemid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mercari-item-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_items(keyword: str, salestatus: str='ON_SALE', pricelow: int=None, itemstatus: str='NEW,ALMOST_NEW', page: int=None, pricehight: int=None, delivery: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "`items` api parameters:
		- keyword: The keyword you want to query
		- page: Which page you want to get (132 item per page)
		- priceLow: Price range start from which price.
		- priceHight: Price range to which price.
		- itemStatus: Item status filter, you can choose one or multiple from
		  ```NEW,ALMOST_NEW,NO_DAMAGES,FEW_DAMAGES,SOME_DAMAGES,BAD_CONDITION```
		- delivery: Delivery type, you can choose one or multiple from
		  ``` CASH_ON_DELIVERY,DELIVERY_COST_INCLUDED```
		- saleStatus: Saling status, you can choose one or multiple from
		  ```ON_SALE,SOLD_OUT```"
    keyword: The keyword you want to query
        salestatus: The item sale status filter, **please don't put any space between STATUS**
        pricelow: The lowest price
        itemstatus: The item statuses filter, **please don't put any space between STATUS**
        page: The page you want to get
        pricehight: The hightest price
        delivery: The delivery types filter, **please don't put any space between DELIVERY_TYPEs**
        
    """
    url = f"https://mercari-item-search.p.rapidapi.com/items"
    querystring = {'keyword': keyword, }
    if salestatus:
        querystring['saleStatus'] = salestatus
    if pricelow:
        querystring['priceLow'] = pricelow
    if itemstatus:
        querystring['itemStatus'] = itemstatus
    if page:
        querystring['page'] = page
    if pricehight:
        querystring['priceHight'] = pricehight
    if delivery:
        querystring['delivery'] = delivery
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mercari-item-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

