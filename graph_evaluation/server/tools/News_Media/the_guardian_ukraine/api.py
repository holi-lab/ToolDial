import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_ukraine_article_of_the_guardian_page_world_ukraine_page(pagenumb: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Ukraine Article of The Guardian page World/Ukraine/{page} (all the pages are available until, beware of the limit)
		Starting at 2"
    
    """
    url = f"https://the-guardian-ukraine.p.rapidapi.com/"
    querystring = {}
    if pagenumb:
        querystring['PageNumb'] = pagenumb
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "the-guardian-ukraine.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

