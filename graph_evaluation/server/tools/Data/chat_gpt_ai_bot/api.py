import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ai_bot(prompt: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Simply ask for whatever you require, and our API will provide the solutions. Let us know your needs, and our AI BOT will effortlessly generate the marketing copy for you. With this convenient tool, you'll have more time to dedicate to your passions."
    
    """
    url = f"https://chat-gpt-ai-bot.p.rapidapi.com/GenerateAIWritter"
    querystring = {'prompt': prompt, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "chat-gpt-ai-bot.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

