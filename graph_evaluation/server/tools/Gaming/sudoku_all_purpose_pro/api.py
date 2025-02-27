import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def create(create: str='32', output: str='raw', width: str=None, quality: str=None, color: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Image output is base64 encoded! See https://myv.at/api/sudoku/code-examples/ how to decode back to HTML, SVG, JPG, PNG, GIF, WEBP easily!"
    output: Use RAW for numbers, HTML for websites or SVG, JPG, PNG, GIF, WEBP to generate images.
        
    """
    url = f"https://sudoku-all-purpose-pro.p.rapidapi.com/sudoku"
    querystring = {}
    if create:
        querystring['create'] = create
    if output:
        querystring['output'] = output
    if width:
        querystring['width'] = width
    if quality:
        querystring['quality'] = quality
    if color:
        querystring['color'] = color
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sudoku-all-purpose-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def solve(quality: str=None, solve: str='000000001000060020901000000710000005000000403000000700000000089000478000060000070', color: str=None, width: str=None, output: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Solve every Sudoku. Simply make an API call with missing digits and receive the answer back immediately. Choose from different output formats to display your Sudoku perfectly."
    
    """
    url = f"https://sudoku-all-purpose-pro.p.rapidapi.com/sudoku"
    querystring = {}
    if quality:
        querystring['quality'] = quality
    if solve:
        querystring['solve'] = solve
    if color:
        querystring['color'] = color
    if width:
        querystring['width'] = width
    if output:
        querystring['output'] = output
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sudoku-all-purpose-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verify(verify: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check any solved (as well as *unsolved!*) Sudoku for authenticity and integrity."
    
    """
    url = f"https://sudoku-all-purpose-pro.p.rapidapi.com/sudoku"
    querystring = {'verify': verify, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "sudoku-all-purpose-pro.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

