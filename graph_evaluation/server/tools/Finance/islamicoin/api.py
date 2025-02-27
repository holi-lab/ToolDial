import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def islamiblockchain_islamicoin_main_cs_txt(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "ISLAMICOIN Circulation Supply"
    
    """
    url = f"https://islamicoin.p.rapidapi.com/ISLAMIBLOCKCHAIN/ISLAMICOIN/main/cs.txt"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "islamicoin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

