import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def download_shopify_order_as_pdf(templatename: str, orderid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Invoice Browse allows merchants to define templates using Liquid or Mustache syntax and export them to pdf using this endpoint."
    orderid: The id of the Shopify order. ID must be provided, order name will not work!
        
    """
    url = f"https://invoice-browse.p.rapidapi.com/orders/{orderid}/{templatename}/pdf"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "invoice-browse.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

