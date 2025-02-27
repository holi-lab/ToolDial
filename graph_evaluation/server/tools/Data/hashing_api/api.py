import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sha_text_hash(datastring: str, mode: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hash text (as query-string-parameter) using SHA1, SHA256, SHA384 or SHA512 algorithm according to selected mode path-parameter [1, 256, 384, 512]."
    
    """
    url = f"https://hashing-api.p.rapidapi.com/api/sha/{mode}/hashtext"
    querystring = {'dataString': datastring, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hashing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def md5_text_hash(datastring: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hash text (as query-string-parameter) using MD5 algorithm."
    
    """
    url = f"https://hashing-api.p.rapidapi.com/api/md5/hashtext"
    querystring = {'dataString': datastring, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hashing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blake2b_text_hash_with_key(datastring: str, keystring: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hash text (as query-string-parameter) with private-key (as query-string-parameter) using blake2b algorithm."
    
    """
    url = f"https://hashing-api.p.rapidapi.com/api/blake2b/hashtextusingkey"
    querystring = {'dataString': datastring, 'keyString': keystring, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hashing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def blake2b_text_hash(datastring: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Hash text (as query-string-parameter) using blake2b algorithm."
    
    """
    url = f"https://hashing-api.p.rapidapi.com/api/blake2b/hashtext"
    querystring = {'dataString': datastring, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hashing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

