import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def isvalidexpression(expression: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    expression: The input expression string.
        
    """
    url = f"https://mathematical-symbolic-expression-manipulator.p.rapidapi.com/IsValidExpression"
    querystring = {'expression': expression, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mathematical-symbolic-expression-manipulator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def evaluate(variables: str, values: str, expression: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    variables: The variables (comma-separated).
        values: The values of the variables (comma-separated).
        expression: The input expression string.
        
    """
    url = f"https://mathematical-symbolic-expression-manipulator.p.rapidapi.com/Evaluate"
    querystring = {'variables': variables, 'values': values, 'expression': expression, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mathematical-symbolic-expression-manipulator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def differentiate(expression: str, variable: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    expression: The input expression string.
        variable: The variable on which to differentiate.
        
    """
    url = f"https://mathematical-symbolic-expression-manipulator.p.rapidapi.com/Differentiate"
    querystring = {'expression': expression, 'variable': variable, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mathematical-symbolic-expression-manipulator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def simplifybasic(expression: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    expression: The input expression string.
        
    """
    url = f"https://mathematical-symbolic-expression-manipulator.p.rapidapi.com/SimplifyBasic"
    querystring = {'expression': expression, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "mathematical-symbolic-expression-manipulator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

