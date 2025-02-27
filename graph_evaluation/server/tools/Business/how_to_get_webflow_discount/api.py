import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def how_to_get_discount_on_webflow(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Webflow is a web-based, design and development tool that is primarily used in developing responsive websites. It comes with a versatile drag-and-drop interface that presents the users with an easy way to create their website.
		
		Webflow provides an array of hosting options for its clients, including AWS Cloud and Google Cloud Platform.
		
		There are many WebFlow discount coupons that can be used to get significant discounts on WebFlow's hosting service. Some of these available on this site:
		
		[Webflow Coupon Code](https://sites.google.com/view/webflow-promo-code)"
    
    """
    url = f"https://how-to-get-webflow-discount.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "how-to-get-webflow-discount.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

