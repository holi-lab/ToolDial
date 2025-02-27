import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def proof(hash: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API endpoint to verify hashes.
		
		###Return values:
		**Hash**: The hash of the file
		**Timestamp**: ISO timestamp in ms
		**identifier**: Optional metadata you have attached
		**messageId**: The message id you can use for verifying the return data"
    
    """
    url = f"https://decentprove.p.rapidapi.com/proof"
    querystring = {'hash': hash, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "decentprove.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

