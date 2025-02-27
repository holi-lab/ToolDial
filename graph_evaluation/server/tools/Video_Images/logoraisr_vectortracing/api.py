import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_download_url_for_result(result_file_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the result image url from the processed image, by passing the `result_file_id` as path parameter in the request url. You get the `result_file_id` as response after creating a project."
    
    """
    url = f"https://logoraisr-vectortracing.p.rapidapi.com/rest-v1/results/{result_file_id}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "logoraisr-vectortracing.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

