import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_company_name_similarity_key_advanced(company: str, algorithm: str, content_type: str='application/json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generates a similarity key for fuzzy matching with other company name records and data"
    
    """
    url = f"https://company-matching-advanced.p.rapidapi.com/getcompanymatchadvanced"
    querystring = {'company': company, 'algorithm': algorithm, }
    if content_type:
        querystring['content-type'] = content_type
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "company-matching-advanced.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

