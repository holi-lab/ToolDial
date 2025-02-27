import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def calling_api_method(method: str, oauth_consumer_key: str, oauth_timestamp: str, oauth_token: str, oauth_signature: str, docid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://edocr-edocr.p.rapidapi.com/api/echo_api"
    querystring = {'method': method, 'oauth_consumer_key': oauth_consumer_key, 'oauth_timestamp': oauth_timestamp, 'oauth_token': oauth_token, 'oauth_signature': oauth_signature, 'docid': docid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "edocr-edocr.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

