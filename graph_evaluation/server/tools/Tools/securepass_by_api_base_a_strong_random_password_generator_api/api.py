import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def generate_password(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The SecurePass API offers a single endpoint ( generate_password )for generating strong and secure passwords. The endpoint is designed to provide developers with an easy-to-use interface for generating custom passwords with different complexity levels, lengths, and character sets."
    
    """
    url = f"https://securepass-by-api-base-a-strong-random-password-generator-api.p.rapidapi.com/generate_password"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "securepass-by-api-base-a-strong-random-password-generator-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

