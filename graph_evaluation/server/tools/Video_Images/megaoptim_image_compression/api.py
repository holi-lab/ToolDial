import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def results_endpoint(proces_id: str, timeout: str='120', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint provides results of the optimization"
    proces_id: The process id returned by the /optimize endpoint initially.
        timeout: The timeout of the result endpoint. It will wait until timeout is reached for result.
        
    """
    url = f"https://megaoptim-image-compression.p.rapidapi.com/optimize/{proces_id}/result"
    querystring = {}
    if timeout:
        querystring['timeout'] = timeout
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "megaoptim-image-compression.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

