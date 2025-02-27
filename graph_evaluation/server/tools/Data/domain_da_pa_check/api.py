import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def url_metrics(target: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "- Target
		- Domain Authority Score
		- Page Authority Score
		- Spam Score
		- Total Backlinks"
    target: Target URL, page or domain
        
    """
    url = f"https://domain-da-pa-check.p.rapidapi.com/"
    querystring = {'target': target, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "domain-da-pa-check.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

