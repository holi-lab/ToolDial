import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getrenderingtaskstatus(taskid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current status of rendering job."
    taskid: Task id of rendering job
        
    """
    url = f"https://web-renderer.p.rapidapi.com/GetRenderingTaskStatus"
    querystring = {'taskId': taskid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-renderer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrenderingtaskstatus(taskid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns current status of rendering job."
    taskid: Task id of rendering job
        
    """
    url = f"https://web-renderer.p.rapidapi.com/GetRenderingTaskStatus"
    querystring = {'taskId': taskid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-renderer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadresult(taskid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads rendered file from server."
    taskid: Task id of rendering task
        
    """
    url = f"https://web-renderer.p.rapidapi.com/DownloadResult"
    querystring = {'taskId': taskid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-renderer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadresult(taskid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Downloads rendered file from server."
    taskid: Task id of rendering task
        
    """
    url = f"https://web-renderer.p.rapidapi.com/DownloadResult"
    querystring = {'taskId': taskid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "web-renderer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

