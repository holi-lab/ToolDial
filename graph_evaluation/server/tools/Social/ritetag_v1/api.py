import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def auto_enhance_post(tweet: str, image: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Analyzes content of a social media post and returns arrays of suggested hashtags, images and final version of enhanced post"
    tweet: text of the post
        image: specify if post already contains a photo (if true, no image will be suggested and the response time will be significantly shorter)
        
    """
    url = f"https://osakasaul-ritetag-v1.p.rapidapi.com/api/v2.2/ai/autoenhance/"
    querystring = {'tweet': tweet, }
    if image:
        querystring['image'] = image
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "osakasaul-ritetag-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def trending_hashtags(green: bool=None, latin: bool=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of up to 20 hashtags trending in the last 24 hours (hourly floating averages)"
    green: restrict results only to green hashtags (currently popular)
        latin: restrict results only to hashtags with latin characters
        
    """
    url = f"https://osakasaul-ritetag-v1.p.rapidapi.com/api/v2.2/data/trending"
    querystring = {}
    if green:
        querystring['green'] = green
    if latin:
        querystring['latin'] = latin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "osakasaul-ritetag-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def influencers_for_hashtag(hashtag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of up to 10 influential Twitter accounts using a specific hashtag"
    
    """
    url = f"https://osakasaul-ritetag-v1.p.rapidapi.com/api/v2.2/data/influencers/{hashtag}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "osakasaul-ritetag-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def historical_data(hashtag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns daily stats of a hashtag for the last 30 days (tweets, retweets, images, links, mentions etc.)"
    
    """
    url = f"https://osakasaul-ritetag-v1.p.rapidapi.com/api/v2.2/data/history/{hashtag}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "osakasaul-ritetag-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def social_media_coach(tweet: str, networks: str, image: int=0, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Analyzes content of a social media post and returns a tailored array of textual tips on how to improve the reach and engagement of that particular post"
    tweet: text of the post
        networks: TWITTER, FACEBOOK, GOOGLE_PLUS
        image: specify if post contains a photo
        
    """
    url = f"https://osakasaul-ritetag-v1.p.rapidapi.com/api/v2.2/ai/coach"
    querystring = {'tweet': tweet, 'networks': networks, }
    if image:
        querystring['image'] = image
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "osakasaul-ritetag-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hashtag_stats(hashtag: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns an array of up to 10 hashtags most frequently used in tweets containing your query + returns stats on 'query hashtag' (hashtag created by adding # sign to your query)."
    
    """
    url = f"https://osakasaul-ritetag-v1.p.rapidapi.com/api/v2.2/data/stats/{hashtag}/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "osakasaul-ritetag-v1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

