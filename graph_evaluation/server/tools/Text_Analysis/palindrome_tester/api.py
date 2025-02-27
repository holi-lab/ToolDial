import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def test_palindrome(palindrome: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The API will respond with a JSON containing two keys: success, and response.
		
		Success will always return a boolean. If there were errors, success will be false, and response will contain an error message string. If there weren't any errors, success will be true, and response will return a boolean: true if the string is a palindrome, and false if it isn't."
    
    """
    url = f"https://palindrome-tester.p.rapidapi.com/{palindrome}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "palindrome-tester.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

