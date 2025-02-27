import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_banned_tags(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "WHAT ARE INSTAGRAM BANNED HASHTAGS?
		Using relevant hashtags on Instagram can boost your engagement and get your Instagram photos in front of many more people. And it also helps Instagram understand what your content is about so it can show it to the right audience.
		
		But not all hashtags are good. Banned hashtags are ones that have been flagged/reported by the Instagram community for a few different reasons.
		
		The first obvious reason is that the hashtag is obscene and people are posting things that are against Instagram’s community guidelines. You will see many of these in our banned hashtags list.
		
		Another reason is that those hashtags have been used by bots and spammers who like to post to those hashtags repeatedly (that’s called SPAM).
		
		So, the reason why you need to avoid using banned hashtags at all, is that Instagram could flag your account as a spammer or as being against their community guidelines.
		
		And this, unfortunately, might result in a shadowban of your account. A shadowban on Instagram means they have limited your account (temporarily), effectively bringing your reach and engagement to almost zero. You might be shadowbanned if you see your hashtags are getting zero engagement, for example."
    
    """
    url = f"https://hashtagy-generate-hashtags.p.rapidapi.com/v1/banned_tags"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hashtagy-generate-hashtags.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_related_hashtags_instagram(keyword: str, include_tags_info: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get related hashtags directly from Instagram."
    keyword: Keyword to get related hashtags
        
    """
    url = f"https://hashtagy-generate-hashtags.p.rapidapi.com/v1/insta/tags"
    querystring = {'keyword': keyword, }
    if include_tags_info:
        querystring['include_tags_info'] = include_tags_info
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hashtagy-generate-hashtags.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_related_hashtags_custom(keyword: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get related hashtags based on custom algorithms"
    keyword: Keyword to get related hashtags
        
    """
    url = f"https://hashtagy-generate-hashtags.p.rapidapi.com/v1/custom_1/tags"
    querystring = {'keyword': keyword, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hashtagy-generate-hashtags.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_comprehensive_hashtags(keyword: str, filter: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a comprehensive list of hashtags based on your keyword"
    keyword: Keyword to get related hashtags
        filter: Type of results (top,random,live). Defaults to: top
        
    """
    url = f"https://hashtagy-generate-hashtags.p.rapidapi.com/v1/comprehensive/tags"
    querystring = {'keyword': keyword, }
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hashtagy-generate-hashtags.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

