import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_job_by_task_id(task_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "you can get the generate job  and  action job status by the task_id, and the task_id will expired at 24 hours after"
    
    """
    url = f"https://midjourney-best-experience.p.rapidapi.com/mj/get-task-id"
    querystring = {'task_id': task_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "midjourney-best-experience.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

