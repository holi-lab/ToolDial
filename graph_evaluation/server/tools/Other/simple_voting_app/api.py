import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def home_page(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is where the main page"
    
    """
    url = f"https://simple-voting-app.p.rapidapi.com/votingapp"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simple-voting-app.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def vote_your_language(languages: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Here the list of programming language : 
		-Java 
		-Python
		-Cplusplus
		-C
		-DotNET
		-JavaScript
		-PHP
		-Swift
		-SQL
		-Ruby
		-Delphi
		-Objective-C
		-Go
		-Assemblylanguage
		-VisualBasic'
		-D
		-R
		-Perl
		-MATLAB"
    
    """
    url = f"https://simple-voting-app.p.rapidapi.com/votingappgetquery"
    querystring = {'languages': languages, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simple-voting-app.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def sort_data(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Sort the language in descending order"
    
    """
    url = f"https://simple-voting-app.p.rapidapi.com/votingappsortdata"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "simple-voting-app.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

