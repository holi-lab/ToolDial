import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def status(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verify if is up"
    
    """
    url = f"https://shopeeapi1.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopeeapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generateshopeelink(x_shopee_appid: str, x_shopee_secret: str, url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/generateShopeeLink?url=SHOPEEURLPRODUCT
		Generates a affiliate link 
		
		Headers
		x-shopee-appId: YOURAPPID
		x-shopee-secret: YOUR SECRET
		
		get this in https://affiliate.shopee.com.br/open_api"
    
    """
    url = f"https://shopeeapi1.p.rapidapi.com/generateShopeeLink"
    querystring = {'x-shopee-appId': x_shopee_appid, 'x-shopee-secret': x_shopee_secret, 'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "shopeeapi1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

