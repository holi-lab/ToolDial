import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getdata(getdata: str=None, getkey: str=None, getcategory: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Extract data"
    
    """
    url = f"https://financialmarketdata.p.rapidapi.com/category"
    querystring = {}
    if getdata:
        querystring['GetData'] = getdata
    if getkey:
        querystring['GetKey'] = getkey
    if getcategory:
        querystring['GetCategory'] = getcategory
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "financialmarketdata.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

