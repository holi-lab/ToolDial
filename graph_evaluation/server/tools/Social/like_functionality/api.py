import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def retrieve_like_data_for_entry(userid: str, postid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will retrieve the # of likes/dislikes towards the given entry via the provided "postId"."
    userid: The user Id to identify which user added the Like/Dislike value towards the entry. This is required and will return a value that will let you know if this user liked, disliked, or did not like/dislike the entry.
        postid: The Id of the entry to retrieve the like/dislike data. 
        
    """
    url = f"https://like-functionality.p.rapidapi.com/like/{postid}"
    querystring = {'userId': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "like-functionality.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

