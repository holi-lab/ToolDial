import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ed_api(grade: str, n: str, text: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "The parameters are:
		text: a description of the content for which to find Common Core State Standards
		grade: which grade(s): k, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 and all.  All means all grades. 
		n: maximum number of matched Common Core State Standards returned, ranked by its relevancy"
    
    """
    url = f"https://ccss_mapping.p.rapidapi.com/cc"
    querystring = {'grade': grade, 'n': n, 'text': text, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ccss_mapping.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

