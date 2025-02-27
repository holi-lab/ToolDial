import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def md5_hash_decrypt(hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Md5 Hash Decrypt "
    hash: Your Hash to Decrypt 
        
    """
    url = f"https://md5-reverse-search.p.rapidapi.com/API/md5.php"
    querystring = {'hash': hash, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "md5-reverse-search.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

