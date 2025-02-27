import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def post_notifications_to_subscribers(url: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "It has two endpoints:
		
		**/subscribe:** This endpoint allows users to subscribe to the website by sending a POST request with their email address in the request body. The email address is then added to a list of subscribers stored by the API.
		
		**/new_post:** This endpoint allows the website to send a notification of a new post to all subscribers. When the website sends a POST request to this endpoint with the new post's title and link in the request body, the API sends an email to each subscriber in the list with a link to the new post using a dummy email service provider."
    
    """
    url = f"https://post-notifications-to-subscribers.p.rapidapi.com/"
    querystring = {}
    if url:
        querystring['url'] = url
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "post-notifications-to-subscribers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

