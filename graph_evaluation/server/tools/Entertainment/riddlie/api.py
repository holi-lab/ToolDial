import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def flag_riddle(is_id: str, flag: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint, you can flag a riddle. This helps the API developer to finetune the API for better service in the future.  You can only use four flags sending the number corresponding to the flag type as below:
		
		0: for flagging the riddle as "Duplicate"
		1: for flagging the riddle as "Inappropriate"
		2: for flagging the riddle as "Not a Riddle"
		3: for flagging the riddle as "Missing Information""
    
    """
    url = f"https://riddlie.p.rapidapi.com/api/v1/riddles/flag/{is_id}"
    querystring = {}
    if flag:
        querystring['flag'] = flag
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "riddlie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def update_level(is_id: str, level: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint enables you to update the level of the Riddle. All the riddles are set to a medium level which is enumerated as "1".  You can suggest the level of a riddle by simply calling this endpoint with query parameters "0", "1", or "2". 
		"0" is Easy
		"1" is Medium 
		"2" is Hard"
    
    """
    url = f"https://riddlie.p.rapidapi.com/api/v1/riddles/level/{is_id}"
    querystring = {}
    if level:
        querystring['level'] = level
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "riddlie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def riddle_by_keyword(keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one random riddle around a specific subject or keyword. For example, a riddle on a "cat"."
    
    """
    url = f"https://riddlie.p.rapidapi.com/api/v1/riddles/bykeyword/{keyword}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "riddlie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def riddle_by_difficulty_level(level: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get one random riddle based on the difficulty. For example, by Easy, Medium, Hard."
    
    """
    url = f"https://riddlie.p.rapidapi.com/api/v1/riddles/bylevel/{level}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "riddlie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def riddle_by_id(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a riddle by ID."
    
    """
    url = f"https://riddlie.p.rapidapi.com/api/v1/riddles/byid/{is_id}"
    querystring = {}
    if is_id:
        querystring['id'] = is_id
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "riddlie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def upvote_riddle(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simply calling this endpoint registers a "like" to better serve the riddles in the future. There is no response back other than a status message."
    
    """
    url = f"https://riddlie.p.rapidapi.com/api/v1/riddles/upvote/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "riddlie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_riddle_of_the_day(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this endpoint, you can request a riddle of the day. Each day API will serve a different riddle with an answer and other details."
    
    """
    url = f"https://riddlie.p.rapidapi.com/api/v1/riddles/rod"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "riddlie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_random_riddle(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This endpoint responds with a new random riddle with an answer and other details."
    
    """
    url = f"https://riddlie.p.rapidapi.com/api/v1/riddles/random"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "riddlie.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

