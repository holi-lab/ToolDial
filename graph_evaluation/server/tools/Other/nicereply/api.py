import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcompanystats(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve all public attributes for given Company identified by API key"
    
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getCompanyStats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuserlist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve list of agents for given Company identified by API key."
    
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getUserList"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcompanyratingscountlastmonth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve number of ratings arrived last month for given Company identified by API key"
    
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getCompanyRatingsCountLastMonth"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuserratingscount(username: str=None, userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve total number of all ratings of given agent for given Company identified by API key."
    username: agent's username, last piece of his link text [use if userid not provided]
        userid: (unsigned int) agent's ID [use if username not provided]
        
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getUserRatingsCount"
    querystring = {}
    if username:
        querystring['username'] = username
    if userid:
        querystring['userid'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpubliccompanyratings(pagesize: str, page: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve public profile ratings for given Company identified by API key."
    pagesize: (unsigned int) number of items per page
        page: (unsigned int) page number of ratings list
        
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getPublicCompanyRatings"
    querystring = {'pagesize': pagesize, }
    if page:
        querystring['page'] = page
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuserdifference(userid: str, username: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve percentage difference of average score of ratings arrived this and previous week for concrete agent within Company identified by API key."
    userid: (unsigned int) agent's ID [use if username not provided]
        username: agent's username, last piece of his link text [use if userid not provided]
        
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getUserDifference"
    querystring = {'userid': userid, }
    if username:
        querystring['username'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcompanyaveragelastmonth(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve average score of ratings arrived last month for given Company identified by API key"
    
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getCompanyAverageLastMonth"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcompanydifference(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve percentage difference of average score of ratings arrived this and previous week for given Company identified by API key"
    
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getCompanyDifference"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcompanyratings(page: str=None, pagesize: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve pagged list of complete ratings of all agents for given Company identified by API key. In reverse time order."
    page: (unsigned int) page number of ratings list
        pagesize: (unsigned int) number of items per page
        
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getCompanyRatings"
    querystring = {}
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pagesize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuserstats(username: str=None, userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve all public attributes of concrete agent within Company identified by API key."
    username: gent's username, last piece of his link text [use if userid not provided]
        userid: (unsigned int) agent's ID [use if username not provided]
        
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getUserStats"
    querystring = {}
    if username:
        querystring['username'] = username
    if userid:
        querystring['userid'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuseraveragelastweek(username: str=None, userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve total number of all ratings of given agent for given Company identified by API key."
    username: agent's username, last piece of his link text [use if userid not provided]
        userid: (unsigned int) agent's ID [use if username not provided]
        
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getUserAverageLastWeek"
    querystring = {}
    if username:
        querystring['username'] = username
    if userid:
        querystring['userid'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcompanyratingscount(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve total number of all ratings for given Company identified by API key"
    
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getCompanyRatingsCount"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuseraveragelastmonth(username: str=None, userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve total number of all ratings of given agent for given Company identified by API key."
    username: agent's username, last piece of his link text [use if userid not provided]
        userid: (unsigned int) agent's ID [use if username not provided]
        
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getUserAverageLastMonth"
    querystring = {}
    if username:
        querystring['username'] = username
    if userid:
        querystring['userid'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcompanyratingscountlastweek(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve number of ratings arrived last week for given Company identified by API key"
    
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getCompanyRatingsCountLastWeek"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuserratingscountlastweek(username: str=None, userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve number of ratings arrived last week for concrete agent within given Company identified by API key."
    username: agent's username, last piece of his link text [use if userid not provided]
        userid: (unsigned int) agent's ID [use if username not provided]
        
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getUserRatingsCountLastWeek"
    querystring = {}
    if username:
        querystring['username'] = username
    if userid:
        querystring['userid'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuserratings(username: str=None, userid: str=None, page: str=None, pagesize: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve pagged list of complete ratings of concrete agent for given Company identified by API key. In reverse time order."
    username: agent's username, last piece of his link text [use if userid not provided]
        userid: (unsigned int) agent's ID [use if username not provided]
        page: (unsigned int) page number of ratings list
        pagesize: (unsigned int) number of items per page
        
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getUserRatings"
    querystring = {}
    if username:
        querystring['username'] = username
    if userid:
        querystring['userid'] = userid
    if page:
        querystring['page'] = page
    if pagesize:
        querystring['pagesize'] = pagesize
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuserratingscountlastmonth(username: str=None, userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve number of ratings arrived last month for concrete agent within given Company identified by API key."
    username: agent's username, last piece of his link text [use if userid not provided]
        userid: (unsigned int) agent's ID [use if username not provided]
        
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getUserRatingsCountLastMonth"
    querystring = {}
    if username:
        querystring['username'] = username
    if userid:
        querystring['userid'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcompanyaverage(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve average score of all ratings for given Company identified by API key"
    
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getCompanyAverage"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getuseraverage(username: str=None, userid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve average score of all ratings of given agent for given Company identified by API key."
    username: agent's username, last piece of his link text [use if userid not provided]
        userid: (unsigned int) agent's ID [use if username not provided]
        
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getUserAverage"
    querystring = {}
    if username:
        querystring['username'] = username
    if userid:
        querystring['userid'] = userid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcompanyaveragelastweek(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "retrieve average score of ratings arrived last week for given Company identified by API key"
    
    """
    url = f"https://nicereply-nicereply.p.rapidapi.com/getCompanyAverageLastWeek"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "nicereply-nicereply.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

