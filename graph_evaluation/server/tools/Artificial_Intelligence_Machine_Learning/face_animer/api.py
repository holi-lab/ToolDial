import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_result(taskid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Enter taskId and get result. Note: you have to wait about 5 seconds after you created the job id. Otherwise response will be Null data."
    taskid: Enter taskId and get result. Note: you have to wait about 5 seconds after you created the job id. Otherwise response will be Null data. 
        
    """
    url = f"https://face-animer.p.rapidapi.com/webFaceDriven/getTaskInfo"
    querystring = {'taskId': taskid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "face-animer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def create_job_id(templateid: str, imageurl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Create Job ID"
    templateid: templateId could be from 0 to 21 (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21)
        imageurl: Enter valid image url
        
    """
    url = f"https://face-animer.p.rapidapi.com/webFaceDriven/submitTaskByUrl"
    querystring = {'templateId': templateid, 'imageUrl': imageurl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "face-animer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

