import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_50m30p3r4t10n(request_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get all the cars in inventory at this endpoint."
    request_id: We use the Request-Id to track the same request across system boundaries.
        
    """
    url = f"https://test-3-spec.p.rapidapi.com/car-inventory"
    querystring = {'Request-Id': request_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "test-3-spec.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

