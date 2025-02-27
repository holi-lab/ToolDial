import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def visualize_get(entry: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Diagram information for visualizing concepts or mind maps."
    entry: Type a word to see the words connected to it
        
    """
    url = f"https://twinword-visual-context-graph.p.rapidapi.com/visualize/"
    querystring = {'entry': entry, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "twinword-visual-context-graph.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

