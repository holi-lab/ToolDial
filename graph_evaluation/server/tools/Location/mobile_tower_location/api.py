import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_location(mnc: int, lac: int, cid: int, mcc: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get location
		
		Params:
		MCC
		MNC
		LAC
		CID"
    
    """
    url = f"https://mobile-tower-location.p.rapidapi.com/location"
    querystring = {'mnc': mnc, 'lac': lac, 'cid': cid, 'mcc': mcc, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mobile-tower-location.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

