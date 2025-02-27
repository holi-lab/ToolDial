import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def query_asynchronous_task_results(job_id: str, type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "For asynchronous interface, after calling the interface, if the real result is not returned; you need to keep the request_id returned by the asynchronous interface and then request this interface to get the real result."
    job_id: Task id：`request_id`
        type: Asynchronous task type.
- `GENERATE_CARTOONIZED_IMAGE`: AI Cartoon Generator.
        
    """
    url = f"https://ai-cartoon-generator.p.rapidapi.com/image/get_async_job_result"
    querystring = {'job_id': job_id, 'type': type, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ai-cartoon-generator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

