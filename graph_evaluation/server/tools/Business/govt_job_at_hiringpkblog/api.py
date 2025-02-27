import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def https_api_citygridmedia_com_content_places_v2_search_where_what_govt_jobs_where_india_ma_page_1_rpp_5_sort_alpha_publisher_test_format_json(hiingpkblog: str, job: str, ggovt: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Latest top best India govt pvt current today new fresh railway, army, Police, teaching, Results, answer key,call letter, state, local,eng.,  vacancy"
    
    """
    url = f"https://govt-job-at-hiringpkblog.p.rapidapi.com/"
    querystring = {'Hiingpkblog': hiingpkblog, 'Job': job, 'GGovt': ggovt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "govt-job-at-hiringpkblog.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

