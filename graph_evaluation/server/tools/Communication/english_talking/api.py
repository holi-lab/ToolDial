import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_an_answer(page: str='1', limit: str='10', status: str='approved', answer: str='Hi, how are you?', is_id: str='5ec47b3d8958430d6a6d5898', speech: str='Hi', user: str='5ec479048958430d6a6d5895', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an answer"
    page: Number of page
        limit: Length of page
        status: approved or analyzing
        answer: Response to the initial speech of the dialogue
        is_id: Unique dialog identifier (automatically generated)


        speech: Speak in which the usuairio wants to get an answer
        user: User who created the dialogue
        
    """
    url = f"https://english-talking2.p.rapidapi.com/v1/dialog"
    querystring = {}
    if page:
        querystring['page'] = page
    if limit:
        querystring['limit'] = limit
    if status:
        querystring['status'] = status
    if answer:
        querystring['answer'] = answer
    if is_id:
        querystring['_id'] = is_id
    if speech:
        querystring['speech'] = speech
    if user:
        querystring['user'] = user
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "english-talking2.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

