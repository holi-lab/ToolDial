import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def listofattributeandvalueoptions(secret: str, e_mail: str, cache_control: str, content_type: str, token: str, action: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Instructions: https://www.ecombr.com/instrucoes-api-i-28.html
		
		Sell on ecombr.com marketplace: https://www.ecombr.com/vender-no-ecombr-i-26.html"
    
    """
    url = f"https://sandbox-ecombr-com-01-products.p.rapidapi.com/api_seller_products_others.php"
    querystring = {'Secret': secret, 'E-mail': e_mail, 'Cache-Control': cache_control, 'Content-Type': content_type, 'Token': token, 'action': action, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sandbox-ecombr-com-01-products.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def listcategoriesandsub_categories(content_type: str, token: str, cache_control: str, secret: str, e_mail: str, action: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Instructions: https://www.ecombr.com/instrucoes-api-i-28.html
		
		Sell on ecombr.com marketplace: https://www.ecombr.com/vender-no-ecombr-i-26.html"
    
    """
    url = f"https://sandbox-ecombr-com-01-products.p.rapidapi.com/api_seller_categories.php"
    querystring = {'Content-Type': content_type, 'Token': token, 'Cache-Control': cache_control, 'Secret': secret, 'E-mail': e_mail, 'action': action, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sandbox-ecombr-com-01-products.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

