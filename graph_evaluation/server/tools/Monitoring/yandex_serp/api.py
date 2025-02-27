import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def gettaskresult_free_of_use(task_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To get the result of the task just provide the **task_id** that you obtained after requesting **GetSerpAsync**. When task is not finished yet you will receive **task_status: inprogress**. For a finished task you will get **task_status: succeeded**. Check the **Example Responses** tab to see the examples. 
		Possible task statuses:
		- **pending** - task is waiting for its turn
		- **inprogress** - task has been passed to the worker
		- **succeeded** - task has been finished with success
		
		Use this endpoint how much you want. **It's free of charge**!
		
		The average **GetCompleteDataAsync**  execution time is 10-40 seconds.  It all depends how many tasks are in the queue."
    
    """
    url = f"https://yandex-serp.p.rapidapi.com/GetTaskResult"
    querystring = {'task_id': task_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yandex-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmytasks_free_of_use(task_status: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves your all tasks. Use this endpoint when lost your **task_id** . Use this endpoint how much you want. **It's free of charge**!"
    
    """
    url = f"https://yandex-serp.p.rapidapi.com/GetMyTasks"
    querystring = {}
    if task_status:
        querystring['task_status'] = task_status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yandex-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_server_time(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns server time."
    
    """
    url = f"https://yandex-serp.p.rapidapi.com/getservertime"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yandex-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def canceltask_free_of_use(task_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cancels pending task. Removes task from your job queue. Use this endpoint how much you want. **It's free of charge**!"
    
    """
    url = f"https://yandex-serp.p.rapidapi.com/CancelTask"
    querystring = {'task_id': task_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yandex-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_serp_async(domain: str, page: int, query: str, lang: str=None, within: str=None, region: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Creates a task that will return complete SERP data for provided domain. Using this endpoint is very simple!
		1. Make a request
		2. Grab received **task_id**
		3. Provide the **task_id** to **GetTaskResult** endpoint to get complete domain data.
		
		The average execution time is 5-40 seconds. It all depends how many tasks are in the queue."
    domain: get search results from:
- yandex.com
- yandex.ru
- yandex.by
- yandex.kz
- yandex.uz
        page: 0 - first page
1 - second page
        lang: enums available:
be - belorussian
de - german
en - english
fr - french
id - indonesian
kk - kazakh
ru - russian
tr - turkish
tt - tatar
uk - ukrainian

e.g. for multiple select: en,fr
        region: e.g. Paris, France
        
    """
    url = f"https://yandex-serp.p.rapidapi.com/GetSerpAsync"
    querystring = {'domain': domain, 'page': page, 'query': query, }
    if lang:
        querystring['lang'] = lang
    if within:
        querystring['within'] = within
    if region:
        querystring['region'] = region
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "yandex-serp.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

