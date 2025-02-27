import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def decryption(decryptionkey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Decrypt any previously encrypted text. You will need the ***encrypted text*** and ***it's matching*** decryption key."
    
    """
    url = f"https://endecrypt.p.rapidapi.com/Decryption/AMCAADMuOr6bA"
    querystring = {'decryptionkey': decryptionkey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "endecrypt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def encryption(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will encrypt any plain text - no punctuation - only spaces are allowed."
    
    """
    url = f"https://endecrypt.p.rapidapi.com/Encryption/texttoencrypt"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "endecrypt.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

