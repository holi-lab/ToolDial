import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_folder_contents(repo: str='gina305/switch-code', path: str='/', key: str='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', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Lists  all of the contents of a specific Github repo."
    
    """
    url = f"https://git-pager.p.rapidapi.com/"
    querystring = {}
    if repo:
        querystring['repo'] = repo
    if path:
        querystring['path'] = path
    if key:
        querystring['key'] = key
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "git-pager.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

