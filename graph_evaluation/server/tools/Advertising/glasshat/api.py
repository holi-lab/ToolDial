import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def client_client_id_get_get_client(client_id: int, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get client"
    client_id: ID of the client to retrieve details.
        apikey: Glasshat API licence key.
        
    """
    url = f"https://brandon-glasshat-glasshat-v1.p.rapidapi.com/client/{client_id}/get"
    querystring = {'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brandon-glasshat-glasshat-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def task_task_id_get_get_action_task(task_id: int, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve details of a specific action task."
    task_id: ID of task action to retrieve details.
        apikey: Glasshat API licence key.
        
    """
    url = f"https://brandon-glasshat-glasshat-v1.p.rapidapi.com/task/{task_id}/get"
    querystring = {'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brandon-glasshat-glasshat-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def project_project_id_get_get_project(project_id: int, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a single project's settings."
    project_id: ID of the project to retrieve details.
        apikey: Glasshat API licence key.
        
    """
    url = f"https://brandon-glasshat-glasshat-v1.p.rapidapi.com/project/{project_id}/get"
    querystring = {'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brandon-glasshat-glasshat-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def client_client_id_projects_get_client_projects(client_id: int, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List all projects belonging to the Glasshat client."
    client_id: ID of the client to retrieve details.
        apikey: Glasshat API licence key.
        
    """
    url = f"https://brandon-glasshat-glasshat-v1.p.rapidapi.com/client/{client_id}/projects"
    querystring = {'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brandon-glasshat-glasshat-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def project_project_id_tasks_get_all_action_tasks(project_id: int, apikey: str, filters: str, with: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List the latest action tasks recommended for the given project."
    project_id: ID of the project to retrieve details.
        apikey: Glasshat API licence key.
        filters: Return health check actions only- (task_type:in:quick_audit:full_audit) Return performance actions only- (task_type:eq:performance) Return performance actions for given keyword only- (task_type:eq:performance,keyword:eq:...) Return performance actions for given URL only- ([ [ task_type, eq, performance ], [ url, eq, http://… ] ])
        with: If string equals 'content', action content will be included in the result. This itself is a nested object.
        
    """
    url = f"https://brandon-glasshat-glasshat-v1.p.rapidapi.com/project/{project_id}/tasks"
    querystring = {'apikey': apikey, 'filters': filters, 'with': with, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "brandon-glasshat-glasshat-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

