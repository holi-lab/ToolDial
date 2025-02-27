import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def jobstatus(authorization: str, x_api_key: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This API is used for getting the status of the job created and getting the final file."
    
    """
    url = f"https://senseipredictpdf.p.rapidapi.com/services/v2/status/gmpf9dnAJrrh8Lj38ipQZqPCMRec98kv"
    querystring = {'Authorization': authorization, 'x-api-key': x_api_key, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "senseipredictpdf.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

