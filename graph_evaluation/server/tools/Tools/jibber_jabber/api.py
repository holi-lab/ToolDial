import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def sentence(minimumwordlength: int=None, numberofwords: int=None, maximumwordlength: int=None, wordlength: int=None, minimumnumberofwords: int=None, maximumnumberofwords: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a single random **sentence**"
    
    """
    url = f"https://jibber-jabber.p.rapidapi.com/sentence"
    querystring = {}
    if minimumwordlength:
        querystring['minimumWordLength'] = minimumwordlength
    if numberofwords:
        querystring['numberOfWords'] = numberofwords
    if maximumwordlength:
        querystring['maximumWordLength'] = maximumwordlength
    if wordlength:
        querystring['wordLength'] = wordlength
    if minimumnumberofwords:
        querystring['minimumNumberOfWords'] = minimumnumberofwords
    if maximumnumberofwords:
        querystring['maximumNumberOfWords'] = maximumnumberofwords
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jibber-jabber.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def paragraph(minimumnumberofwords: int=None, wordlength: int=None, maximumnumberofwords: int=None, numberofsentences: int=None, minimumnumberofsentences: int=None, maximumwordlength: int=None, maximumnumberofsentences: int=None, numberofwords: int=None, minimumwordlength: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a **paragraph** with random sentences"
    
    """
    url = f"https://jibber-jabber.p.rapidapi.com/paragraph"
    querystring = {}
    if minimumnumberofwords:
        querystring['minimumNumberOfWords'] = minimumnumberofwords
    if wordlength:
        querystring['wordLength'] = wordlength
    if maximumnumberofwords:
        querystring['maximumNumberOfWords'] = maximumnumberofwords
    if numberofsentences:
        querystring['numberOfSentences'] = numberofsentences
    if minimumnumberofsentences:
        querystring['minimumNumberOfSentences'] = minimumnumberofsentences
    if maximumwordlength:
        querystring['maximumWordLength'] = maximumwordlength
    if maximumnumberofsentences:
        querystring['maximumNumberOfSentences'] = maximumnumberofsentences
    if numberofwords:
        querystring['numberOfWords'] = numberofwords
    if minimumwordlength:
        querystring['minimumWordLength'] = minimumwordlength
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jibber-jabber.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def words(minimumwordlength: int=None, maximumwordlength: int=None, numberofwords: int=None, wordlength: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns random **words**"
    
    """
    url = f"https://jibber-jabber.p.rapidapi.com/words"
    querystring = {}
    if minimumwordlength:
        querystring['minimumWordLength'] = minimumwordlength
    if maximumwordlength:
        querystring['maximumWordLength'] = maximumwordlength
    if numberofwords:
        querystring['numberOfWords'] = numberofwords
    if wordlength:
        querystring['wordLength'] = wordlength
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jibber-jabber.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sentences(maximumnumberofsentences: int=None, minimumwordlength: int=None, minimumnumberofsentences: int=None, numberofsentences: int=None, maximumnumberofwords: int=None, minimumnumberofwords: int=None, numberofwords: int=None, wordlength: int=None, maximumwordlength: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns some single random **sentence**"
    
    """
    url = f"https://jibber-jabber.p.rapidapi.com/sentences"
    querystring = {}
    if maximumnumberofsentences:
        querystring['maximumNumberOfSentences'] = maximumnumberofsentences
    if minimumwordlength:
        querystring['minimumWordLength'] = minimumwordlength
    if minimumnumberofsentences:
        querystring['minimumNumberOfSentences'] = minimumnumberofsentences
    if numberofsentences:
        querystring['numberOfSentences'] = numberofsentences
    if maximumnumberofwords:
        querystring['maximumNumberOfWords'] = maximumnumberofwords
    if minimumnumberofwords:
        querystring['minimumNumberOfWords'] = minimumnumberofwords
    if numberofwords:
        querystring['numberOfWords'] = numberofwords
    if wordlength:
        querystring['wordLength'] = wordlength
    if maximumwordlength:
        querystring['maximumWordLength'] = maximumwordlength
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jibber-jabber.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def word(minimumwordlength: int=None, maximumwordlength: int=None, wordlength: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a random **word**"
    
    """
    url = f"https://jibber-jabber.p.rapidapi.com/word"
    querystring = {}
    if minimumwordlength:
        querystring['minimumWordLength'] = minimumwordlength
    if maximumwordlength:
        querystring['maximumWordLength'] = maximumwordlength
    if wordlength:
        querystring['wordLength'] = wordlength
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "jibber-jabber.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

