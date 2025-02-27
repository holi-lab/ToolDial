import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_coverage(postcode: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get cell coverage for AT&T, T-Mobile, U.S. Cellular and Verizon by zip code"
    
    """
    url = f"https://us-cell-coverage-by-zip-code.p.rapidapi.com/"
    querystring = {'postcode': postcode, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "us-cell-coverage-by-zip-code.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

