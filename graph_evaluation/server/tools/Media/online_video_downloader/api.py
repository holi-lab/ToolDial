import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def fetch_task_status_and_get_download_link(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "/tasks/{id} pass the id you get when you call create download task api...
		once you create the download task use this api to fetch the status of the download link generation.
		Checkout the **About** section for api docs.
		All steps will be found in the **About** section of api."
    
    """
    url = f"https://online-video-downloader1.p.rapidapi.com/tasks/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "online-video-downloader1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

