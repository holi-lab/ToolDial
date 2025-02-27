import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def query_asynchronous_task_results(type: str, job_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For asynchronous interface, after calling the interface, if the real result is not returned; you need to keep the request_id returned by the asynchronous interface and then request this interface to get the real result."
    type: Asynchronous task type.
- `HD_COLOUR_MIGRATION`: HD color migration.
        job_id: Task id：`request_id`
        
    """
    url = f"https://hd-colour-migration.p.rapidapi.com/image/get_async_job_result"
    querystring = {'type': type, 'job_id': job_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hd-colour-migration.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

