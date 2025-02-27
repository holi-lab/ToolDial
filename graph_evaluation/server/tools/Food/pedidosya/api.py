import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_restaurant_by_link(country: str, link: str, city: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Restaurant by Link"
    country: Allowed countries:
ar,cl,cr,ec,sv,gt,hn,ni,pa,py,pe,do,ve,uy
        
    """
    url = f"https://pedidosya.p.rapidapi.com/"
    querystring = {'country': country, 'link': link, 'city': city, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pedidosya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_restaurants_by_address(address: str, country: str, businesstype: str, limit: int=10, offset: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Restaurants by Address"
    country: Allowed countries:
ar,cl,cr,ec,sv,gt,hn,ni,pa,py,pe,do,ve,uy
        
    """
    url = f"https://pedidosya.p.rapidapi.com/address"
    querystring = {'address': address, 'country': country, 'businessType': businesstype, }
    if limit:
        querystring['limit'] = limit
    if offset:
        querystring['offset'] = offset
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pedidosya.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

