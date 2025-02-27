import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def detokenize(token: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The detokenize end point takes a token and returns the original string provided for that token.  The original string was provided in the tokenize endpoint call, provided as the "value" parameter."
    
    """
    url = f"https://tokenizer4.p.rapidapi.com/detokenize"
    querystring = {'token': token, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokenizer4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tokenize(value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tokenizes a value passed to the endpoint.  The value must be 1KB or smaller and the follow characters are not allowed:
		- The forward slash (/) character
		- The backslash (\) character
		- The number sign (#) character
		- The question mark (?) character
		- Control characters from U+0000 to U+001F, including:
		     - The horizontal tab (\t) character
		     - The linefeed (\n) character
		     - The carriage return (\r) character
		- Control characters from U+007F to U+009F"
    
    """
    url = f"https://tokenizer4.p.rapidapi.com/tokenize"
    querystring = {'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tokenizer4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

