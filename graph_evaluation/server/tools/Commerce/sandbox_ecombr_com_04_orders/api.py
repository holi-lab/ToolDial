import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def listoforders(token: str, cache_control: str, e_mail: str, content_type: str, secret: str, status: str, action: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Instructions: https://www.ecombr.com/instrucoes-api-i-28.html
		
		Sell on ecombr.com marketplace: https://www.ecombr.com/vender-no-ecombr-i-26.html"
    
    """
    url = f"https://sandbox-ecombr-com-04-orders.p.rapidapi.com/api_seller_orders.php"
    querystring = {'Token': token, 'Cache-Control': cache_control, 'E-mail': e_mail, 'Content-Type': content_type, 'Secret': secret, 'status': status, 'action': action, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sandbox-ecombr-com-04-orders.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getthelistofcountriesregistered(action: str, content_type: str, cache_control: str, secret: str, token: str, e_mail: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Instructions: https://www.ecombr.com/instrucoes-api-i-28.html
		
		Sell on ecombr.com marketplace: https://www.ecombr.com/vender-no-ecombr-i-26.html"
    
    """
    url = f"https://sandbox-ecombr-com-04-orders.p.rapidapi.com/api_seller_products_others.php"
    querystring = {'action': action, 'Content-Type': content_type, 'Cache-Control': cache_control, 'Secret': secret, 'Token': token, 'E-mail': e_mail, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sandbox-ecombr-com-04-orders.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

