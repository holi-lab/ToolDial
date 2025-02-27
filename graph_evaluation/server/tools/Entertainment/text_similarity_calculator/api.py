import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getpercentage(ftext: str, stext: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Calculate the percentage of match between two texts."
    ftext: Enter text number one
        stext: Enter text number two.
        
    """
    url = f"https://text-similarity-calculator.p.rapidapi.com/stringcalculator.php"
    querystring = {'ftext': ftext, 'stext': stext, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "text-similarity-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

