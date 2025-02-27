import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def jobs(title: str='python developer', post_date: str='2023-05-17', place: str='munich', company: str='facebook', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can query Python developer jobs by giving your parameters."
    
    """
    url = f"https://linkedin-jobs-python-developer.p.rapidapi.com/jobs/"
    querystring = {}
    if title:
        querystring['title'] = title
    if post_date:
        querystring['post_date'] = post_date
    if place:
        querystring['place'] = place
    if company:
        querystring['company'] = company
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkedin-jobs-python-developer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def main(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Linkedin Python Developer Jobs API"
    
    """
    url = f"https://linkedin-jobs-python-developer.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "linkedin-jobs-python-developer.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

