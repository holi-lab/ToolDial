import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def search_users(kapi_proxy: str=None, country: str='ma', pcursor: str=None, language: str='en', user_name: str='paul', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Users API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/search-users"
    querystring = {}
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if country:
        querystring['country'] = country
    if pcursor:
        querystring['pcursor'] = pcursor
    if language:
        querystring['language'] = language
    if user_name:
        querystring['user_name'] = user_name
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_posts(keyword: str, kapi_proxy: str=None, pcursor: str=None, language: str='en', country: str='ma', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Posts API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/search-posts"
    querystring = {'keyword': keyword, }
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if pcursor:
        querystring['pcursor'] = pcursor
    if language:
        querystring['language'] = language
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_trending(kapi_proxy: str=None, language: str='en', pcursor: str=None, country: str='ma', count: str='30', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Trending API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/search-trending"
    querystring = {}
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if language:
        querystring['language'] = language
    if pcursor:
        querystring['pcursor'] = pcursor
    if country:
        querystring['country'] = country
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_top_music(kapi_proxy: str=None, country: str='ma', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Top Music API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/search-top-music"
    querystring = {}
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_recommend(kapi_proxy: str=None, language: str='en', pcursor: str=None, country: str='ma', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Recommend API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/user-recommend"
    querystring = {}
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if language:
        querystring['language'] = language
    if pcursor:
        querystring['pcursor'] = pcursor
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def login_with_mobile_code(countrycode: str, sms_code: str, session: str, mobile: str, kapi_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Login with Mobile Code API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/login-with-mobile-code"
    querystring = {'countryCode': countrycode, 'sms_code': sms_code, 'session': session, 'mobile': mobile, }
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def send_sms_code(countrycode: str, mobile: str, kapi_proxy: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Send SMS Code API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/send-sms-code"
    querystring = {'countryCode': countrycode, 'mobile': mobile, }
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_followers(userid: str, token: str, kapi_proxy: str=None, pcursor: str=None, count: str='20', page: str='1', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Followers API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/user-followers"
    querystring = {'userId': userid, 'token': token, }
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if pcursor:
        querystring['pcursor'] = pcursor
    if count:
        querystring['count'] = count
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_following(userid: str, token: str, kapi_proxy: str=None, page: str='1', pcursor: str=None, count: str='20', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Following API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/user-following"
    querystring = {'userId': userid, 'token': token, }
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if page:
        querystring['page'] = page
    if pcursor:
        querystring['pcursor'] = pcursor
    if count:
        querystring['count'] = count
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_user(userid: str, kapi_proxy: str=None, language: str='en', country: str='ma', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get User API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/user-profile"
    querystring = {'userId': userid, }
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if language:
        querystring['language'] = language
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def tag_search(keyword: str, kapi_proxy: str=None, count: str='30', language: str='en', country: str='ma', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Tag Search API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/tag-search"
    querystring = {'keyword': keyword, }
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if count:
        querystring['count'] = count
    if language:
        querystring['language'] = language
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_mix(keyword: str, kapi_proxy: str=None, pcursor: str=None, country: str='ma', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Mix API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/search-mix"
    querystring = {'keyword': keyword, }
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if pcursor:
        querystring['pcursor'] = pcursor
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_suggest(keyword: str, kapi_proxy: str=None, country: str='ma', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Suggest API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/search-suggest"
    querystring = {'keyword': keyword, }
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def search_music(keyword: str, kapi_proxy: str=None, country: str='ma', pcursor: str=None, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search Music API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/search-music"
    querystring = {'keyword': keyword, }
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if country:
        querystring['country'] = country
    if pcursor:
        querystring['pcursor'] = pcursor
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_music(musicid: str, kapi_proxy: str=None, country: str='ma', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Music API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/get-music"
    querystring = {'musicId': musicid, }
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def top_music(kapi_proxy: str=None, count: str='20', pcursor: str=None, language: str='en', country: str='ma', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Top Music API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/top-music"
    querystring = {}
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if count:
        querystring['count'] = count
    if pcursor:
        querystring['pcursor'] = pcursor
    if language:
        querystring['language'] = language
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def liked_posts(userid: str, kapi_proxy: str=None, language: str='en', count: str='30', country: str='ma', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Liked Posts API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/liked-posts"
    querystring = {'userId': userid, }
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if language:
        querystring['language'] = language
    if count:
        querystring['count'] = count
    if country:
        querystring['country'] = country
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_feed(userid: str, kapi_proxy: str=None, country: str='ma', count: str='30', pcursor: str=None, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "User Feed API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/feed-profile"
    querystring = {'userId': userid, }
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if country:
        querystring['country'] = country
    if count:
        querystring['count'] = count
    if pcursor:
        querystring['pcursor'] = pcursor
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feed_hot(kapi_proxy: str=None, country: str='ma', count: str='30', language: str='en', pcursor: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Feed Hot API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/feed-hot"
    querystring = {}
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if country:
        querystring['country'] = country
    if count:
        querystring['count'] = count
    if language:
        querystring['language'] = language
    if pcursor:
        querystring['pcursor'] = pcursor
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feed(kapi_proxy: str=None, country: str='ma', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Feed API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/feed"
    querystring = {}
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def list_comments(photoid: str, kapi_proxy: str=None, count: str='10', order: str='desc', country: str='ma', pcursor: str=None, language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List Comments API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/list-comments"
    querystring = {'photoId': photoid, }
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if count:
        querystring['count'] = count
    if order:
        querystring['order'] = order
    if country:
        querystring['country'] = country
    if pcursor:
        querystring['pcursor'] = pcursor
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_post(photoid: str, kapi_proxy: str=None, country: str='ma', language: str='en', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get Post API"
    
    """
    url = f"https://kwai4.p.rapidapi.com/get-post"
    querystring = {'photoId': photoid, }
    if kapi_proxy:
        querystring['kapi-proxy'] = kapi_proxy
    if country:
        querystring['country'] = country
    if language:
        querystring['language'] = language
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "kwai4.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

