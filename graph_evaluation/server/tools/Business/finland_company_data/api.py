import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def searchbyid(businessid: str, maxresults: int, resultsfrom: int, totalresults: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search by Finnish Company ID, the format must be in the form "0112038-9" (Nokia)."
    
    """
    url = f"https://finland-company-data.p.rapidapi.com/bis/v1"
    querystring = {'businessId': businessid, 'maxResults': maxresults, 'resultsFrom': resultsfrom, 'totalResults': totalresults, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "finland-company-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

