import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product_details(store_id: int, tcin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns single product details
		
		- Every target has a uniq **store_id** , **store_id** can be extract with the help of this endpoint **/target/location/search**"
    store_id: Store id

- Every target has a uniq **stored_id** , **stored_id** can be extract with the help of this endpoint **/target/location/search**
        tcin: Product id

For example: 53331580

**/product/search** will contain **tcin** for each product in the list
        
    """
    url = f"https://target-com-store-product-reviews-locations-data.p.rapidapi.com/product/details"
    querystring = {'store_id': store_id, 'tcin': tcin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target-com-store-product-reviews-locations-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_reviews(tcin: str, limit: int=100, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns product reviews"
    tcin: Product id

For example: 53331580

**/product/search** will contain **tcin** for each product in the list
        limit: Limit the output number of records. 

- Default is 100
- Max number is 100

        offset: Skin ~~n~~ amount of records

Default: 0
        
    """
    url = f"https://target-com-store-product-reviews-locations-data.p.rapidapi.com/product/reviews"
    querystring = {'tcin': tcin, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target-com-store-product-reviews-locations-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def product_search(store_id: int, keyword: str, limit: int=24, offset: int=0, rating: int=0, sponsored: int=1, sort_by: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns products from the Target.Com Search Result
		
		- Every target has a uniq **store_id** , **store_id** can be extract with the help of this endpoint **/target/location/search**"
    store_id: Store id

- Every target has a uniq **store_id** , **store_id** can be extract with the help of this endpoint **/target/location/search**
        keyword: Search keyword

For example: **iphone case**
        limit: Limit the output number of records. 

- Default is 24
- Max number is 24

        offset: Skin ~~n~~ amount of records

Default: 0
        rating: To show only products with the rating for example >=4 you can set query value to 4 . It can be 1,2,3,4,5
        sponsored: Should sponsored products be included in the result
        sort_by: You can sort products by using this query:

- Relevance: **relevance**
- Featured: **featured**
- Price low to high: **pricelow**
- Price high to low: **pricehigh**
- Best selling: **bestselling**
- Newest products: **newest**

For example if you need to sort search by \\\"Price low to high\\\" then you would need to set query value to **pricelow**

        
    """
    url = f"https://target-com-store-product-reviews-locations-data.p.rapidapi.com/product/search"
    querystring = {'store_id': store_id, 'keyword': keyword, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    if rating:
        querystring['rating'] = rating
    if sponsored:
        querystring['sponsored'] = sponsored
    if sort_by:
        querystring['sort_by'] = sort_by
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target-com-store-product-reviews-locations-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def target_store_location_search(zip: int, radius: str='100', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The endpoint returns Target Store locations details
		
		- Only US **zip** codes are accepted
		- **radius** is in miles"
    zip: US 5 digit zip code

For example: 11203
        radius: Radius of search

Radius is in **miles**

For example: 100
        
    """
    url = f"https://target-com-store-product-reviews-locations-data.p.rapidapi.com/location/search"
    querystring = {'zip': zip, }
    if radius:
        querystring['radius'] = radius
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "target-com-store-product-reviews-locations-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

