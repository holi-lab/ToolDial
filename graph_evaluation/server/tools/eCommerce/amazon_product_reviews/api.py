import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def amazonproductreviews(pagenum: int, domain: str, asin: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It will accept three parameters asin, domain, pageNum.
		For Ex:
		asin: B087N3LWRB  (you can find it in the product page of amazon)
		domain: amazon.co.uk  (put the domain without www, currently we only provide amazon.com, amazon.in and amazon.co.uk)
		pageNum: 1 (min 1 and max depends on the number of reviews on the product.  Each page give 10 reviews)"
    
    """
    url = f"https://amazon-product-reviews1.p.rapidapi.com/amazonreviews/"
    querystring = {'pageNum': pagenum, 'domain': domain, 'asin': asin, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-product-reviews1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

