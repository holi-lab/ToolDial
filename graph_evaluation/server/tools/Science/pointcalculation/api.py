import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def translate_point(angle: int, distance: int, y1: int, x1: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Moves point a given distance and angle"
    angle: angle in radians (eg 4)
        distance: distance (eg 5)
        y1: y1 (eg 0)
        x1: x1 (eg 0)
        
    """
    url = f"https://pointcalculation.p.rapidapi.com/translate-point"
    querystring = {'angle': angle, 'distance': distance, 'y1': y1, 'x1': x1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pointcalculation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def distance(x1: int, x2: int, y2: int, y1: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Finds the distance between two points"
    x1: x1 (eg 2)
        x2: x2 (eg 2)
        y2: y2 (eg 4)
        y1: y1 (eg 2)
        
    """
    url = f"https://pointcalculation.p.rapidapi.com/distance"
    querystring = {'x1': x1, 'x2': x2, 'y2': y2, 'y1': y1, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pointcalculation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def rotate_point(y2: int, y1: int, x1: int, x2: int, angle: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Rotates a point around another point by a given angle."
    y2: y2 (eg 0)
        y1: y1 (eg 0)
        x1: x1 (eg 5)
        x2: x2 (eg 0)
        angle: angle in radians (eg 6)
        
    """
    url = f"https://pointcalculation.p.rapidapi.com/rotate-point"
    querystring = {'y2': y2, 'y1': y1, 'x1': x1, 'x2': x2, 'angle': angle, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pointcalculation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def find_point_on_line(x1: int, distance_from_start: int, x2: int, y1: int, y2: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a point between to points as a given distance"
    x1: x1 (eg 0)
        distance_from_start: distance from start (eg 3)
        x2: x2 (eg 3)
        y1: y1 (eg 0)
        y2: y2 (eg 4)
        
    """
    url = f"https://pointcalculation.p.rapidapi.com/find-point-on-line"
    querystring = {'x1': x1, 'distance_from_start': distance_from_start, 'x2': x2, 'y1': y1, 'y2': y2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pointcalculation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def angle(y1: int, y2: int, x1: int, x2: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Finds the angle of the line that passes through both of the points. Angle is returned in Radians."
    y1: y1 (eg 37)
        y2: y2 (eg -51)
        x1: x1 (eg 30)
        x2: x2 (eg 79)
        
    """
    url = f"https://pointcalculation.p.rapidapi.com/angle"
    querystring = {'y1': y1, 'y2': y2, 'x1': x1, 'x2': x2, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pointcalculation.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

