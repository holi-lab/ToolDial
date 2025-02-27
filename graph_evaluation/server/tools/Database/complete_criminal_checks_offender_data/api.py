import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def api_docs(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Easily create your links!"
    
    """
    url = f"https://complete-criminal-checks-offender-data.p.rapidapi.com/completecriminalchecks.com/api/v2/xml/?DataBase=7&state=ALL&firstname=George&lastname=Zimmerman&exact=0&aka=0&limit=100"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "complete-criminal-checks-offender-data.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

