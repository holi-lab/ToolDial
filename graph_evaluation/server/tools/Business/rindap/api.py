import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def list_workspaces(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rindap (Process) Workspace is a customizable place enabling you to create your Tasks, Workers, TaskQueues and Workflows elements in it to access and manage your business processes. The elements defined in one Workspace are specific to it and cannot be shared with other Workspace.
		
		Various tasks with different attributes, specific and modifiable workflow, various skilled workers can be defined in one Workspace to orchestrate your business processes efficiently in the lowest time. Business Process Workspace helps organizations to improve their internal processes management by providing a single container to manage, control and monitor the whole elements needed in one place. With Rindap, managing the inner operational processes of a company in a purpose-built Workspace and defining your strategies and policies for the company in a single Workspace can be done."
    
    """
    url = f"https://rindap.p.rapidapi.com/Workspaces"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rindap.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def fetch_single_workspace(workspacesid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "FETCH Single Workspace"
    
    """
    url = f"https://rindap.p.rapidapi.com/Workspaces/{workspacesid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "rindap.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

