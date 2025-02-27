import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_response(job_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With the JOB ID generated on the "Rewriter" endpoint, you will be able to get the response from that rewrite."
    job_id: The JOB ID generated on "Rewriter" endpoint.
        
    """
    url = f"https://best-paraphrasing-api.p.rapidapi.com/retrieve"
    querystring = {'job_id': job_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "best-paraphrasing-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

