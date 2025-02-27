import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_task(request_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To get the response of the API request when used in asynchronous mode, you can use either of the following methods:
		
		Pre-configuring a web-hook URL with IDfy Whenever the submitted task is completed, the response for the task would be sent back to your application through a POST HTTP request.
		
		Making a GET call to IDfy GET call can be made to IDfy to fetch the API response for the completed tasks. You can create a GET call using the request_id."
    
    """
    url = f"https://pan-card-ocr1.p.rapidapi.com/v3/tasks"
    querystring = {'request_id': request_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pan-card-ocr1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

