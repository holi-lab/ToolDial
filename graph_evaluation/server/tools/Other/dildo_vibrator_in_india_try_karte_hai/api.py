import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def dildo_vibrator_in_india(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "When you are considering getting a [dildo vibrator in india](https://trykartehai.com/collections/dildo-in-india/), you are opening the doors to enhance your sexual repertoire. Trykartehai opens the door for you to experiment with all your sensual fantasies. We are one of the most trusted adult stores in India for premium products."
    
    """
    url = f"https://dildo-vibrator-in-india-try-karte-hai.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "dildo-vibrator-in-india-try-karte-hai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

