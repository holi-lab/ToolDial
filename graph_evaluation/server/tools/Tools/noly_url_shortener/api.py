import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def shorten_long_link(link: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Use this endpoint to shorten a URL. The response will be a JSON object with the following structure:
		
		On success:
		{
		"status": 1,
		"error": "",
		"data": "https://tombo.icu/57h80n"
		}
		
		On failure:
		{
		"status": 0,
		"error": "This website is already shortened here.",
		"data": "https://tombo.icu/57h80n"
		}
		
		The "data" field contains the shortened URL."
    
    """
    url = f"https://noly-url-shortener.p.rapidapi.com/shorten/{link}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "noly-url-shortener.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

