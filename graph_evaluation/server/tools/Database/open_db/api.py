import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def https_web_cts_000webhostapp_com_console_api_v1_create(p1: str, dn: str, v1: str, is_id: str, p2: str='user_email', v2: str='example@example.com', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "THIS ENDPOINT IS USED TO STORE YOUR DATA INSIDE THE OPEN DATABSE { YOU HAVE TO SET CUSTOM NAME FOR YOUR DATABasE }
		REMEMber this name as it will be used to fetch data from databse ."
    
    """
    url = f"https://open-db1.p.rapidapi.com/"
    querystring = {'p1': p1, 'dn': dn, 'v1': v1, 'id': is_id, }
    if p2:
        querystring['p2'] = p2
    if v2:
        querystring['v2'] = v2
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "open-db1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

