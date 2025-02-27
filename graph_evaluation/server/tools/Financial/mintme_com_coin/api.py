import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def webchain(webchain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "MintMe.com Coin is a transparent web-mineable blockchain platform"
    webchain: MintMe.com Coin is a transparent web-mineable blockchain platform
        
    """
    url = f"https://mintme-com-coin.p.rapidapi.com/coin"
    querystring = {'Webchain': webchain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mintme-com-coin.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

