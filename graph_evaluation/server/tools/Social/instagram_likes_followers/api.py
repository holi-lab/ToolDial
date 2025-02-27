import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def check_order_status(order: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check the status on an order."
    
    """
    url = f"https://instagram-likes-followers.p.rapidapi.com/status.php"
    querystring = {'order': order, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-likes-followers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_likes(amount: int, picture_link: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "generate likes for a picture or video.  Likes are sent to the object within 10 minutes. In exceptional cases, delivery can take a little longer.
		The account that posted the picture/video needs to be public while the order is processed!"
    picture_link: The link to the picture the likes should be sent to. Need to be in the form https://www.instagram.com/p/XXX/
        
    """
    url = f"https://instagram-likes-followers.p.rapidapi.com/likes.php"
    querystring = {'amount': amount, 'picture_link': picture_link, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-likes-followers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_followers(amount: int, username: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send followers to an account. Followers are delivered within around 10 minutes. In exceptional cases, delivery can take a little longer. Account needs to be PUBLIC while the order is processed."
    username: The username of the account the followers should be sent to. Account must be public while the order is processed.
        
    """
    url = f"https://instagram-likes-followers.p.rapidapi.com/followers.php"
    querystring = {'amount': amount, 'username': username, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "instagram-likes-followers.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

