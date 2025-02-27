import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def product(locale: str, asins: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Amazon product price information by locale.
		
		ASINs provided will be searched by the locale provided, e.g. if `B005YQZ1KE,B074R8RQQ2` were provided and `US` as a locale were provided, it will provide metadata from only `amazon.com`.
		
		Please note that price data may be up to 30 minutes out of date.
		
		Currently supported locales:
		
		US (amazon.com)
		UK (amazon.co.uk)
		CA (amazon.ca)
		IN (amazon.in)
		FR (amazon.fr)
		
		More coming soon!"
    locale: Currently supported locales:

US (amazon.com)
UK (amazon.co.uk)
CA (amazon.ca)
IN (amazon.in)
FR (amazon.fr)

More coming soon!
        
    """
    url = f"https://amazon-product-price-data.p.rapidapi.com/product"
    querystring = {'locale': locale, 'asins': asins, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "amazon-product-price-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

