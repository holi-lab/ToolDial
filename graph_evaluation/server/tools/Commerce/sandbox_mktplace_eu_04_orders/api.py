import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getthelistofcountriesregistered(secret: str, content_type: str, token: str, e_mail: str, cache_control: str, action: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Instructions: https://www.mktplace.eu/api-instructions-i-32.html
		
		Sell on mktplace.eu: https://www.mktplace.eu/sell-on-mktplace-i-25.html"
    
    """
    url = f"https://sandbox-mktplace-eu-04-orders.p.rapidapi.com/api_seller_products_others.php"
    querystring = {'Secret': secret, 'Content-Type': content_type, 'Token': token, 'E-mail': e_mail, 'Cache-Control': cache_control, 'action': action, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sandbox-mktplace-eu-04-orders.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listoforders(cache_control: str, content_type: str, secret: str, token: str, e_mail: str, action: str, status: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Instructions: https://www.mktplace.eu/api-instructions-i-32.html
		
		Sell on mktplace.eu: https://www.mktplace.eu/sell-on-mktplace-i-25.html"
    
    """
    url = f"https://sandbox-mktplace-eu-04-orders.p.rapidapi.com/api_seller_orders.php"
    querystring = {'Cache-Control': cache_control, 'Content-Type': content_type, 'Secret': secret, 'Token': token, 'E-mail': e_mail, 'action': action, 'status': status, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sandbox-mktplace-eu-04-orders.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

