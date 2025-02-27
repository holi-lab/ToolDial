import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def have_a_chat_with_our_bot(question: str, chathistory: str="What is Dua to recite when you see a miraculous thing like Islam & AI? | There is no specific dua (supplication) mentioned in the Quran or Hadith that is to be recited when witnessing a miracle. However, Muslims are encouraged to constantly remember Allah and express their gratitude to Him for the blessings and miracles they witness in their lives. In general, it is recommended to say 'Subhan Allah' (Glory be to Allah) when witnessing something amazing or miraculous", toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Have a conversation with our Islam&AI Bot in which it remembers your previous conversation! (upto 30 texts)"
    
    """
    url = f"https://islam-ai-api.p.rapidapi.com/api/chat"
    querystring = {'question': question, }
    if chathistory:
        querystring['chatHistory'] = chathistory
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "islam-ai-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_an_answer_to_your_question(question: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get answer to an Islamic question from our Islam&AI bot"
    
    """
    url = f"https://islam-ai-api.p.rapidapi.com/api/bot"
    querystring = {'question': question, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "islam-ai-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

