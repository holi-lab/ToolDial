import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def index_factor(country: str, index_base: int, index_year_month: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "In case of any problem or question contact me under ch.walter@gmx.at 
		
		Structure of the query URL (example):
		
		https://index.smartapicloud.com/id/DE/2015/202012
		
		DE - the country is entered here (DE for Germany or AT for Austria or CH for Swiss)
		2015 - the base is entered here (2015 or 2020)
		202012 - the year and month are entered here in the format YYYYMM (2020 (year) 12 (the month December))
		
		The result is a JSON string:
		
		{“value”: “109.40000”, “more”: [{“index_base”: “VPI2015”, “index_month”: “202012”, “index_type”: “TOTAL INDEX”}]}"
    
    """
    url = f"https://consumer-price-index-germany-austria-swiss.p.rapidapi.com/{country}/{index_base}/{index_year_month}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "consumer-price-index-germany-austria-swiss.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

