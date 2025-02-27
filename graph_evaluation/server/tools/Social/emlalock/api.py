import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def add(apikey: str, userid: str, value: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will raise your duration by VALUE."
    apikey: Your API key.
        userid: Your UserID.
        value: Time value (in seconds or short terms).
        
    """
    url = f"https://emlalock.p.rapidapi.com/add"
    querystring = {'apikey': apikey, 'userid': userid, 'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addmaximum(value: str, userid: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will raise your maximum duration by VALUE."
    value: Time value (in seconds or short terms).
        userid: Your UserID.
        apikey: Your API key.
        
    """
    url = f"https://emlalock.p.rapidapi.com/addmaximum"
    querystring = {'value': value, 'userid': userid, 'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addmaximumrandom(to: str, is_from: str, userid: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will raise your maximum duration randomly between FROM and TO."
    to: Time value (in seconds or short terms).
        from: Time value (in seconds or short terms).
        userid: Your UserID.
        apikey: Your API key.
        
    """
    url = f"https://emlalock.p.rapidapi.com/addmaximumrandom"
    querystring = {'to': to, 'from': is_from, 'userid': userid, 'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addminimum(userid: str, value: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will raise your minimum duration by VALUE."
    userid: Your UserID.
        value: Time value (in seconds or short terms).
        apikey: Your API key.
        
    """
    url = f"https://emlalock.p.rapidapi.com/addminimum"
    querystring = {'userid': userid, 'value': value, 'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addminimumrandom(to: str, userid: str, is_from: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will raise your minimum duration randomly between FROM and TO."
    to: Time value (in seconds or short terms).
        userid: Your UserID.
        from: Time value (in seconds or short terms).
        apikey: Your API key.
        
    """
    url = f"https://emlalock.p.rapidapi.com/addminimumrandom"
    querystring = {'to': to, 'userid': userid, 'from': is_from, 'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addrandom(apikey: str, is_from: str, to: str, userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will raise your duration randomly between FROM and TO."
    apikey: Your API key.
        from: Time value (in seconds or short terms).
        to: Time value (in seconds or short terms).
        userid: Your UserID.
        
    """
    url = f"https://emlalock.p.rapidapi.com/addrandom"
    querystring = {'apikey': apikey, 'from': is_from, 'to': to, 'userid': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addrequirement(apikey: str, userid: str, value: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will raise your requirements by VALUE."
    apikey: Your API key.
        userid: Your UserID.
        value: Number of requirements.
        
    """
    url = f"https://emlalock.p.rapidapi.com/addrequirement"
    querystring = {'apikey': apikey, 'userid': userid, 'value': value, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def addrequirementrandom(apikey: str, to: int, is_from: int, userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will raise your requirements randomly between FROM and TO."
    apikey: Your API key.
        to: Number of requirements.
        from: Number of requirements.
        userid: Your UserID.
        
    """
    url = f"https://emlalock.p.rapidapi.com/addrequirementrandom"
    querystring = {'apikey': apikey, 'to': to, 'from': is_from, 'userid': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def info(userid: str, apikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will do nothing except give you the information each API-job does."
    userid: Your UserID.
        apikey: Your API key.
        
    """
    url = f"https://emlalock.p.rapidapi.com/info"
    querystring = {'userid': userid, 'apikey': apikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sub(holderapikey: str, value: str, userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will lower your duration by VALUE."
    holderapikey: The API key of the holder.
        value: Time value (in seconds or short terms).
        userid: Your UserID.
        apikey: Your API key.
        
    """
    url = f"https://emlalock.p.rapidapi.com/sub"
    querystring = {'holderapikey': holderapikey, 'value': value, 'userid': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def submaximum(holderapikey: str, value: str, userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will lower your maximum duration by VALUE."
    holderapikey: The API key of the holder.
        value: Time value (in seconds or short terms).
        userid: Your UserID.
        apikey: Your API key.
        
    """
    url = f"https://emlalock.p.rapidapi.com/submaximum"
    querystring = {'holderapikey': holderapikey, 'value': value, 'userid': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def submaximumrandom(userid: str, apikey: str, is_from: str, to: str, holderapikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will lower your maximum duration randomly between FROM and TO."
    userid: Your UserID.
        apikey: Your API key.
        from: Time value (in seconds or short terms).
        to: Time value (in seconds or short terms).
        holderapikey: The API key of the holder.
        
    """
    url = f"https://emlalock.p.rapidapi.com/submaximumrandom"
    querystring = {'userid': userid, 'apikey': apikey, 'from': is_from, 'to': to, 'holderapikey': holderapikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subminimum(apikey: str, holderapikey: str, value: str, userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will lower your minimum duration by VALUE."
    apikey: Your API key.
        holderapikey: The API key of the holder.
        value: Time value (in seconds or short terms).
        userid: Your UserID.
        
    """
    url = f"https://emlalock.p.rapidapi.com/subminimum"
    querystring = {'apikey': apikey, 'holderapikey': holderapikey, 'value': value, 'userid': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subminimumrandom(apikey: str, to: str, holderapikey: str, userid: str, is_from: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will lower your minimum duration randomly between FROM and TO."
    apikey: Your API key.
        to: Time value (in seconds or short terms).
        holderapikey: The API key of the holder.
        userid: Your UserID.
        from: Time value (in seconds or short terms).
        
    """
    url = f"https://emlalock.p.rapidapi.com/subminimumrandom"
    querystring = {'apikey': apikey, 'to': to, 'holderapikey': holderapikey, 'userid': userid, 'from': is_from, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subrandom(to: str, apikey: str, holderapikey: str, is_from: str, userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will lower your duration randomly between FROM and TO."
    to: Time value (in seconds or short terms).
        apikey: Your API key.
        holderapikey: The API key of the holder.
        from: Time value (in seconds or short terms).
        userid: Your UserID.
        
    """
    url = f"https://emlalock.p.rapidapi.com/subrandom"
    querystring = {'to': to, 'apikey': apikey, 'holderapikey': holderapikey, 'from': is_from, 'userid': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subrequirement(value: int, apikey: str, userid: str, holderapikey: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will lower your maximum duration by VALUE."
    value: Number of requirements.
        apikey: Your API key.
        userid: Your UserID.
        holderapikey: The API key of the holder.
        
    """
    url = f"https://emlalock.p.rapidapi.com/subrequirement"
    querystring = {'value': value, 'apikey': apikey, 'userid': userid, 'holderapikey': holderapikey, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def subrequirementrandom(apikey: str, holderapikey: str, userid: str, is_from: int, to: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This Request will lower your minimum duration randomly between FROM and TO."
    apikey: Your API key.
        holderapikey: The API key of the holder.
        userid: Your UserID.
        from: Number of requirements.
        to: Number of requirements.
        
    """
    url = f"https://emlalock.p.rapidapi.com/subrequirementrandom"
    querystring = {'apikey': apikey, 'holderapikey': holderapikey, 'userid': userid, 'from': is_from, 'to': to, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user(sessionid: str, userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about user or yourseld"
    sessionid: returned from Login endpoint
        userid: UserID or yourself for own data (includes more data than for other users)
        
    """
    url = f"https://emlalock.p.rapidapi.com/user"
    querystring = {'sessionid': sessionid, 'userid': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def feed(sessionid: str, chastitysessionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the activity feed for a given session."
    sessionid: returned from Login endpoint
        chastitysessionid: SessionID of the chastity session
        
    """
    url = f"https://emlalock.p.rapidapi.com/feed"
    querystring = {'sessionid': sessionid, 'chastitysessionid': chastitysessionid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def session(sessionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get your session timing information."
    sessionid: returned from Login endpoint
        
    """
    url = f"https://emlalock.p.rapidapi.com/session/"
    querystring = {'sessionid': sessionid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages(sessionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a list of your received and send private messages."
    sessionid: returned from Login endpoint
        
    """
    url = f"https://emlalock.p.rapidapi.com/messages"
    querystring = {'sessionid': sessionid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def chastity_session(sessionid: str, chastitysessionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a chastity session by their sessionID."
    sessionid: returned from Login endpoint
        chastitysessionid: sessionID or default
        
    """
    url = f"https://emlalock.p.rapidapi.com/chastitysession"
    querystring = {'sessionid': sessionid, 'chastitysessionid': chastitysessionid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def user_chastity_session(sessionid: str, userid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a users session by their userID"
    sessionid: returned from Login endpoint
        sessionid: returned from Login endpoint
        userid: UserID or yourself for own data
        
    """
    url = f"https://emlalock.p.rapidapi.com/userchastitysession"
    querystring = {'sessionid': sessionid, 'userid': userid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def messages_info(sessionid: str, messageid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get information about a specific message by messageID."
    sessionid: returned from Login endpoint
        messageid: MessageID of the message to show.
        
    """
    url = f"https://emlalock.p.rapidapi.com/messages"
    querystring = {'sessionid': sessionid, 'messageid': messageid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def keys(sessionid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "List your wearers chastity sessions"
    sessionid: returned from Login endpoint
        
    """
    url = f"https://emlalock.p.rapidapi.com/keys"
    querystring = {'sessionid': sessionid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "emlalock.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

