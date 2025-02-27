import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def request_results_for_the_tasks(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this call to retrieve results from the requests (getItems, getSearch, getProduct)"
    id: Id that was returned from making request calls
        
    """
    url = f"https://sku-io.p.rapidapi.com/result/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sku-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def request_list_of_items_for_specific_search_keyword_from_the_store(function: str, store: str, param: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve list of items for any search keyword.
		
		Supported stores and their respective codes can be [checked using this api call](https://whichstore.skuio.com/v1/store?url=http://amazon.com).
		
		Information returned: list of products, next page url, list of child categories, list of root categories.
		
		Products usually have this information (sometimes there are extra data as well, depends on the store):** title, image, price, url**"
    function: Function used to make the request. For this call only getItems is available. 
        store: Store code, see a list of supported stores [here](https://skugrid.com/remoteApi/?supportedStores)
        param: Keyword that will be used to perform the search
        
    """
    url = f"https://sku-io.p.rapidapi.com/request"
    querystring = {'function': function, 'store': store, 'param': param, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sku-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def request_full_product_information(param: str, store: str, function: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Request a product information (title, description, prices, stock, variations, etc.) for the url or for the sku. 
		
		Supported stores and their respective codes can be [checked using this api call](https://whichstore.skuio.com/v1/store?url=http://amazon.com).
		
		Depending on the store some extra information can be returned, but the basic information is always returned, it includes, but not limited to:
		
		title
		description
		images
		product_id
		priceList
		stockAvailability
		offers (if more than one)
		shippingCost
		shippingDays (# of days before the product is shipped)
		variations (if present)
		features
		specifications"
    param: it can be a sku or full url
        store: Store code, see a list of supported stores [here](https://skugrid.com/remoteApi/?supportedStores)
        function: Function used to make the request. For this call only getProduct is available. 
        
    """
    url = f"https://sku-io.p.rapidapi.com/request"
    querystring = {'param': param, 'store': store, 'function': function, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sku-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def request_list_of_items_from_specific_page_of_store_s_catalog(function: str, store: str, param: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve list of items from any catalog url of the store.
		
		Supported stores and their respective codes can be [checked using this api call](https://whichstore.skuio.com/v1/store?url=http://amazon.com).
		
		Information returned: list of products, next page url, list of child categories, list of root categories.
		
		Products usually have this information (sometimes there are extra data as well, depends on the store):** title, image, price, url**"
    function: Function used to make the request. For this call only getItems is available. 
        store: Store code, see a list of supported stores [here](https://skugrid.com/remoteApi/?supportedStores)
        param: Url that points to a specific catalog page of the store
        
    """
    url = f"https://sku-io.p.rapidapi.com/request"
    querystring = {'function': function, 'store': store, 'param': param, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sku-io.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

