import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def wtp_pricing_using_fips(fips: str, adj_range: int, base_price: int, precision: str='99', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given the base_price (minimum price) , adjustment range, and FIPS, and an optional precision, it provides a dynamic price.
		
		example: given a price of $100.00, an adjustment range of $20.00, and a FIPS code, it will give a dynamic price between $100.00 and $120.00.
		
		note: please provide floating point numbers for base_price and adj_range.
		
		To format the price such that it ends with specific digits, you can use the optional precision parameter. This parameter is a string that indicates the last n digits. for example, if the final adjusted price is 122.02, given "99" as the precision,  it will make the price 121.99."
    
    """
    url = f"https://wtp-dynamic-pricing.p.rapidapi.com/optimize/fips/{base_price}/{adj_range}/{fips}/{precision}"
    querystring = {}
    if precision:
        querystring['precision'] = precision
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wtp-dynamic-pricing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def wtp_pricing_using_zip_code(base_price: int, zipcode: str, adj_range: int, precision: str='99', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Given the base_price (minimum price) , adjustment range, and zip code, and an optional precision, it provides a dynamic price.
		
		example: given a price of $100.00, an adjustment range of $20.00, and a zip code, it will give a dynamic price between $100.00 and $120.00.
		
		note: please provide floating point numbers for base_price and adj_range.
		
		To format the price such that it ends with specific digits, you can use the optional precision parameter. This parameter is a string that indicates the last n digits. for example, if the final adjusted price is 122.02, given "99" as the precision,  it will make the price 121.99."
    
    """
    url = f"https://wtp-dynamic-pricing.p.rapidapi.com/optimize/zipcode/{base_price}/{adj_range}/{zipcode}/{precision}"
    querystring = {}
    if precision:
        querystring['precision'] = precision
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "wtp-dynamic-pricing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

