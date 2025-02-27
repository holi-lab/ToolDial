import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def personyze_rest(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Personyze External API makes possible for third-party applications to read and change data on your real account, like add/delete Placeholders, Products in the catalog, etc.
		
		The API utilizes HTTP protocol. If your application is capable of making HTTP requests (nowadays this is true for almost 100% applications), it can connect to Personyze server, send HTTP requests and get responces.
		
		Data exchange is happening in JSON format."
    
    """
    url = f"https://personyze-rest.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "personyze-rest.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

