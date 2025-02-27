import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_job_by_taskid(taskid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "fetch result of job by taskId, and the taskId will expired after 24 hours
		
		- taskId (required,string)
		
		![](https://i.ibb.co/pZ6rNcR/bc9062aa-8580-4170-9bfa-cd3c4cafd481-1893x1615.jpg)"
    
    """
    url = f"https://midjourney-api5.p.rapidapi.com/task/"
    querystring = {'taskId': taskid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "midjourney-api5.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

