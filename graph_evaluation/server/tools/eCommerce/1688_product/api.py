import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_item_detail(item_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get item detail by item_id,
		If the product url like this: https://detail.1688.com/offer/653761459790.html?spm=a260k.dacugeneral.home2019rec.4.6633436cMBEbie&&scm=1007.21237.114566.0&pvid=573aebf8-5008-4134-a254-22282e2893b3&object_id=653761459790&scm2=1007.30657.177495.0&pvid2=7f4ebb02-e13b-494f-ad0b-a3db9579eb01&udsPoolId=2274586&resourceId=1797996&resultType=normal,
		then the item_id is 653761459790."
    item_id: If the product url like this: https://detail.1688.com/offer/653761459790.html?spm=a260k.dacugeneral.home2019rec.4.6633436cMBEbie&&scm=1007.21237.114566.0&pvid=573aebf8-5008-4134-a254-22282e2893b3&object_id=653761459790&scm2=1007.30657.177495.0&pvid2=7f4ebb02-e13b-494f-ad0b-a3db9579eb01&udsPoolId=2274586&resourceId=1797996&resultType=normal,
then the item_id is 653761459790.
        
    """
    url = f"https://1688-product2.p.rapidapi.com/api/sc/1688/v2/item_detail"
    querystring = {'item_id': item_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "1688-product2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_item_description(item_id: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get item description by item_id,return the description images and html of the product"
    
    """
    url = f"https://1688-product2.p.rapidapi.com/api/sc/1688/item_desc"
    querystring = {'item_id': item_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "1688-product2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

