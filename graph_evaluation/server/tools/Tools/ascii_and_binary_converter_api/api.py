import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def binary(words: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes a 'words' parameter as a string and converts it to binary. It uses the GET method and returns the result in json format. The input is passed through some validation before processing it, and it also uses caching mechanism to improve performance."
    
    """
    url = f"https://ascii-and-binary-converter-api.p.rapidapi.com/binary"
    querystring = {'words': words, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ascii-and-binary-converter-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def ascii(words: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint takes a 'words' parameter as a string and converts it to ASCII. It uses the GET method and returns the result in json format. The input is passed through some validation before processing it, and it also uses caching mechanism to improve performance."
    
    """
    url = f"https://ascii-and-binary-converter-api.p.rapidapi.com/ascii"
    querystring = {'words': words, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ascii-and-binary-converter-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

