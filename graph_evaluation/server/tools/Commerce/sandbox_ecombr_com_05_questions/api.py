import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getthelistofquestionsordoubtsfromcustomersinterestedintheproducts(e_mail: str, token: str, cache_control: str, secret: str, content_type: str, action: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API Instructions: https://www.ecombr.com/instrucoes-api-i-28.html
		
		Sell on ecombr.com marketplace: https://www.ecombr.com/vender-no-ecombr-i-26.html"
    
    """
    url = f"https://sandbox-ecombr-com-05-questions.p.rapidapi.com/api_seller_products_others.php"
    querystring = {'E-mail': e_mail, 'Token': token, 'Cache-Control': cache_control, 'Secret': secret, 'Content-Type': content_type, 'action': action, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sandbox-ecombr-com-05-questions.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

