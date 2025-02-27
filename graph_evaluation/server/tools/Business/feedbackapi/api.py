import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def health(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://feedbackapi.p.rapidapi.com/health"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feedbackapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfeedbacks(page: str='[
  "/docs/page-name"
]', reasons: str='[
  "solved_my_problem"
]', sentiment: str='[]', apioperationid: str='[
  "getUser"
]', userid: str='[
  "abc"
]', rating: str='[]', userip: str='[
  "91.201.190.10"
]', category: str='[
  "Support"
]', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    page: Page to filter by.
        reasons: Reasons to filter by.
        sentiment: Sentiment to filter by.
        apioperationid: Operation to filter by.
        userid: User ID to filter by.
        rating: Rating to filter by.
        userip: IP address to filter by.
        category: Category to filter by.
        
    """
    url = f"https://feedbackapi.p.rapidapi.com/feedback"
    querystring = {}
    if page:
        querystring['page'] = page
    if reasons:
        querystring['reasons'] = reasons
    if sentiment:
        querystring['sentiment'] = sentiment
    if apioperationid:
        querystring['apiOperationId'] = apioperationid
    if userid:
        querystring['userId'] = userid
    if rating:
        querystring['rating'] = rating
    if userip:
        querystring['userIP'] = userip
    if category:
        querystring['category'] = category
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feedbackapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfeedback(is_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    id: Resource identifier string.
        
    """
    url = f"https://feedbackapi.p.rapidapi.com/feedback/{is_id}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "feedbackapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

