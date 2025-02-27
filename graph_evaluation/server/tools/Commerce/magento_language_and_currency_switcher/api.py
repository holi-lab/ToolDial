import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def http_www_magentocommerce_com_magento_connect_geo_ip_language_currency_switcher_html(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Automatic Language and Currency Switcher Magento Extension"
    
    """
    url = f"https://simonwalker-magento-language-and-currency-switcher-v1.p.rapidapi.com/http://www.magentocommerce.com/magento-connect/geo-ip-language-currency-switcher.html"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simonwalker-magento-language-and-currency-switcher-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

