import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search(search: str, authorization: str='Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOlwvXC9hcGkuaW9kb2pvLmNvbVwvdG9rZW4iLCJpYXQiOjE2Mzk0OTc0MjksIm5iZiI6MTYzOTQ5NzQyOSwianRpIjoiT3NXRnRoTlN2UnBkSXpqNyIsInN1YiI6MSwicHJ2IjoiODdlMGFmMWVmOWZkMTU4MTJmZGVjOTcxNTNhMTRlMGIwNDc1NDZhYSJ9.TbOkiefndtSZ2spS11fTeR8jPtVPv9UVKg7xrmueff0', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Find information based on either code or description."
    
    """
    url = f"https://icd10.p.rapidapi.com/icd10/v1/{search}"
    querystring = {}
    if authorization:
        querystring['Authorization'] = authorization
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "icd10.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

