import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_industry_similarity(code2: str, code1: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns the numeric similarity between two given industries (i.e., their NACE industry codes)
		Supported Industries and their codes can be found here: https://ec.europa.eu/competition/mergers/cases/index/nace_all.html"
    
    """
    url = f"https://nace-industry-similarities.p.rapidapi.com/getNaceIndustrySimilarity"
    querystring = {'code2': code2, 'code1': code1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nace-industry-similarities.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

