import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def tokens(network: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "tokens list and contract based on network parameter."
    network: arbitrum | aurora | avalanche | bsc | cronos | ethereum | fantom | harmony | heco | kcc | metis | moonriver | oasisemerald | optimism | polygon | telos
        
    """
    url = f"https://tokenlist.p.rapidapi.com/tokens/{network}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokenlist.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

