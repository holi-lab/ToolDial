import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def synonyms(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    ""
    
    """
    url = f"https://aqeel-mashkraft-touchbase-v1.p.rapidapi.com/words/hello/synonyms?accessToken=j4d04qyNZGmshUdUIjqOJWoG3AuZp1tt0zxjsn2KQE1Wg0esvG"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aqeel-mashkraft-touchbase-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

