import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def random(depth: int=3, length: int=3, maxlength: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simple GET request that would return 5 randomly generated JSON objects with pseudo-random structure."
    depth: Provides the maximum depth of the JSON object. This is the number of levels the JSON object.
Defaults to 3.
For depth=3, the structure would look like 
{
   {
      {
      }
   }
}
        length: This parameter represents the desired number of generated JSON objects in the response. 
Possible values are [1-5]. Anything greater than 5 will be ceilled to 5.
Defaults to 5.
        maxlength: This is the maximum number of fields per level in the JSON object.
Defaults to 10.
        
    """
    url = f"https://dummy-json-generator.p.rapidapi.com/random"
    querystring = {}
    if depth:
        querystring['depth'] = depth
    if length:
        querystring['length'] = length
    if maxlength:
        querystring['maxLength'] = maxlength
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dummy-json-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

