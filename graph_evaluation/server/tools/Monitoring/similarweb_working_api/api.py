import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcompletedata_async_creates_a_task_to_process(domain: str, callback_url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "👁️‍🗨️Read [5 easy steps to use ASYNC](https://rapidapi.com/letsscrape/api/similarweb-working-api/tutorials/how-to-use-async-endpoints-to-get-complete-data%3F)
		Creates a task that will return complete domain data from SimilarWeb when finished. Using this endpoint is very simple!
		1. Make a request providing **domain** name
		2. Grab received **task_id**
		3. Provide the **task_id** to **GetTaskResult** endpoint to get complete domain data.
		
		The average execution time is 10-40 seconds. It all depends how many tasks are in the queue.
		You individual queue could can contain up to 5 tasks. Need more? Contact us!"
    callback_url: After finishing your task the system will send the result to provided **callback_url**. System sends result in **json** format by using **POST** method. If you have more questions about it write us! To get result in php just use:
*$result = file_get_contents('php://input');*
        
    """
    url = f"https://similarweb-working-api.p.rapidapi.com/similarweb/GetCompleteDataAsync"
    querystring = {'domain': domain, }
    if callback_url:
        querystring['callback_url'] = callback_url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "similarweb-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_additional_domain_data_bonus_try_it(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets additional domain information like:
		- Backlinks report
		- SEMrush data
		- MOZ data
		- Revenue/Loss report
		- Search engine data
		- and more!"
    
    """
    url = f"https://similarweb-working-api.p.rapidapi.com/similarweb/GetAdditionalDomainData"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "similarweb-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_domain_data(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Gets **basic** domain data from SimilarWeb. You will receive from this endpoint data like: top country shares, monthly visits, user engagements, ranks and traffic sources. In many cases such data **meet the clients needs**. If you want complete domain data check **GetCompleteDataAsync** out!"
    
    """
    url = f"https://similarweb-working-api.p.rapidapi.com/similarweb/getdata"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "similarweb-working-api.p.rapidapi.com"
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
    url = f"https://similarweb-working-api.p.rapidapi.com/similarweb/CancelTask"
    querystring = {'task_id': task_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "similarweb-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_server_time(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns server time"
    
    """
    url = f"https://similarweb-working-api.p.rapidapi.com/similarweb/GetServerTime"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "similarweb-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettaskresult_free_of_use(task_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "To get the result of the task just provide the **task_id** that you obtained after requesting **GetCompleteDataAsync**. When task is not finished yet you will receive **task_status: inprogress**. For a finished task you will get **task_status: succeeded**. Check the **Example Responses** tab to see the examples. 
		Possible task statuses:
		- **pending** - task is waiting for its turn
		- **inprogress** - task has been passed to the worker
		- **succeeded** - task has been finished with success
		
		Use this endpoint how much you want. **It's free of charge**!
		
		The average **GetCompleteDataAsync**  execution time is 10-40 seconds.  It all depends how many tasks are in the queue."
    task_id: To get the **task_id** call the **GetCompleteDataAsync**
        
    """
    url = f"https://similarweb-working-api.p.rapidapi.com/similarweb/GetTaskResult"
    querystring = {'task_id': task_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "similarweb-working-api.p.rapidapi.com"
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
    url = f"https://similarweb-working-api.p.rapidapi.com/similarweb/GetMyTasks"
    querystring = {}
    if task_status:
        querystring['task_status'] = task_status
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "similarweb-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_complete_data_obsolete_use_async_version_instead(domain: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "⚠️**OBSOLETE! Use GetCompleteDataASYNC version instead.**
		
		Detailed data retrieval time usually takes 5-120 seconds.  **Unfortunately**, RapidAPI timeout limit is 60 seconds, so not all requests will be succeed."
    
    """
    url = f"https://similarweb-working-api.p.rapidapi.com/similarweb/getfulldata"
    querystring = {'domain': domain, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "similarweb-working-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

