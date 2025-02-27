import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_transcript(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a transcript by job id"
    id: Gets an existing transcription by job id
        
    """
    url = f"https://rev-ai.p.rapidapi.com/jobs/{is_id}/transcript"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rev-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jobs_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets an existing transcription job by id"
    id: Gets an existing transcription job by id
        
    """
    url = f"https://rev-ai.p.rapidapi.com/jobs/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rev-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jobs_list(starting_after: str=None, limit: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets a list of transcription jobs submitted within the last week in reverse chronological order up to limit jobs per call. Pagination is supported via passing the last job id from a previous call into starting_after."
    starting_after: Gets jobs listed after this job id, exclusive - job with id not included
        limit: Limits the number of jobs returned, default is 100, max is 1000
        
    """
    url = f"https://rev-ai.p.rapidapi.com/jobs"
    querystring = {}
    if starting_after:
        querystring['starting_after'] = starting_after
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rev-ai.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

