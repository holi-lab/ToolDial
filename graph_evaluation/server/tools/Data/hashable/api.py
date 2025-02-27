import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def hash_strings(method: str, str: str, encoding: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint is the main and the one used for hashing strings.
		This endpoint requires to have 3 query parameters with the request.
		
		They are:
		- str: the string to hash
		- method: the method to use like "MD5, "SHA256, "SHA512".
		- encoding: the type of encoding to use to encode the hash.
		
		A sample request would look like this:
		```js
		fetch('https://hashable-api.herokuapp.com/hash?str=hello&method=sha256&encoding=hex')
		.then(res => res)
		.catch(err => console.error(err))
		```"
    
    """
    url = f"https://hashable.p.rapidapi.com/hash/{str}/{method}/{encoding}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hashable.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

