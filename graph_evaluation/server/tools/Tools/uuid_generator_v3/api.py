import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def uuid_version_version(version: int, type: str=None, text: str=None, count: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random UUID (v4)."
    version: Version of the UUID spec to use. (0-5), Use '0' for nil (00000000-0000-0000-0000-000000000000) UUID.
        type: For v3 and v5 of UUID Spec you can supply the type (dns/url/oid/x500/nil).
        text: For v3 and v5 of UUID Spec supply the text value for the type specified dns/url/oid/x500/nil. For example specify a dns/domain string if the type is "dns"
        count: Number of UUID's to generate (defaults to 1)
        
    """
    url = f"https://uuid-generator2.p.rapidapi.com/uuid/version/{version}"
    querystring = {}
    if type:
        querystring['type'] = type
    if text:
        querystring['text'] = text
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uuid-generator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def uuid(count: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generate a random UUID (v4)."
    count: Number of UUID's to generate (defaults to 1)
        
    """
    url = f"https://uuid-generator2.p.rapidapi.com/uuid"
    querystring = {}
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "uuid-generator2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

