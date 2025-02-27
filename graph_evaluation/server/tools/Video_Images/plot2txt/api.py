import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def cspan_video_search(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "A demo for the wav2txt API method, showing various chunks of searchable speech from a CSPAN video. JSON output is base64 encoded."
    
    """
    url = f"https://plot2txt1.p.rapidapi.com/cspan-video-search"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "plot2txt1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def arxiv_figure_search(label: str, source: str, skey: str=None, ymax: str=None, xmin: str=None, xmax: str=None, ymin: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Search and retrieve arxiv figure data eg,. labels, data ranges and x-y points. This example was created using the simple scatter plot API method, output is JSON, with links to detected plots and output QC images showing centers of fitted Gaussians."
    label: Combined x and y axis text labels
        source: The arxiv journal source eg., astro
        skey: The start time for next returned query object
        ymax: Upper bound for y axis values
        xmin: Lower bound for x axis values
        xmax: Upper bound for x axis values
        ymin: Lower bound for y axis values
        
    """
    url = f"https://plot2txt1.p.rapidapi.com/arxiv-figure-search"
    querystring = {'label': label, 'source': source, }
    if skey:
        querystring['skey'] = skey
    if ymax:
        querystring['ymax'] = ymax
    if xmin:
        querystring['xmin'] = xmin
    if xmax:
        querystring['xmax'] = xmax
    if ymin:
        querystring['ymin'] = ymin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "plot2txt1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

