import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_dad_jokes(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random dad joke from a collection of over 20,000 jokes. The jokes are curated to be family-friendly and appropriate for all ages. The response is returned in JSON format with the joke text included as a string. Users can call this endpoint as many times as they like to receive a new joke each time."
    
    """
    url = f"https://dad-jokes21.p.rapidapi.com/dad"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dad-jokes21.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

