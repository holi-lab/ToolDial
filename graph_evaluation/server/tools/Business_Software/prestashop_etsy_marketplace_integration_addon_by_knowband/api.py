import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def prestashop_etsy_marketplace_integration_addon_knowband(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Etsy is one of the top successful Marketplace running globally. With a million of consumers visiting the website on a daily basis, Etsy Marketplace enables the sellers to expand their store reach and boost their sale by showcasing their goods at the Marketplace. Knowbands [PrestaShop Etsy Connector Addon](url=https://www.knowband.com/prestashop-etsy-marketplace-integration) helps the admin to start selling on the Etsy Marketplace. Using the Etsy Marketplace Integrator, store admin can save time by reducing the manual listing time. Prestashop Etsy Marketplace Integration Addon allows the store owner to handle the inventory and order from the backend of the PrestaShop store."
    
    """
    url = f"https://prestashop-etsy-marketplace-integration-addon-by-knowband.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "prestashop-etsy-marketplace-integration-addon-by-knowband.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

