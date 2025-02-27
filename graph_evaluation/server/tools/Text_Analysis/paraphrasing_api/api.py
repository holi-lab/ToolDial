import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_response(job_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With the JOB ID generated on the "Paraphrase" endpoint, you will be able to get the response from that rewrite."
    job_id: The JOB ID generated on \"Paraphrase\" endpoint.
        
    """
    url = f"https://paraphrasing-api2.p.rapidapi.com/retrieve"
    querystring = {'job_id': job_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "paraphrasing-api2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

