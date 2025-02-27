import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def store_locator(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Magento 2 store locator extension by Knowband](https://www.knowband.com/magento-2-store-locator-and-pickup)"
    
    """
    url = f"https://magento-2-store-locator-extension-by-knowband.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magento-2-store-locator-extension-by-knowband.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def store_locator_copy(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "[Magento 2 store locator extension by Knowband](https://www.knowband.com/magento-2-store-locator-and-pickup)"
    
    """
    url = f"https://magento-2-store-locator-extension-by-knowband.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "magento-2-store-locator-extension-by-knowband.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

