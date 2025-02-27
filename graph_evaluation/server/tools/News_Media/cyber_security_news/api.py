import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def this_will_get_cy_ber_security_news_from_specific_new_outlet(newspaperid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "News outlets we have currently are 
		
		1. cnbc
		2. wired
		3. threatpost
		4. bbc
		5. reuters
		6. latimes
		7. smh
		8. un
		9. es
		10. nyp
		11. wsj
		12. thn
		13. guardian
		14. abc
		15. cyberinsiders
		
		**Some news outlets will not show any response because if no new news**"
    
    """
    url = f"https://cyber-security-news.p.rapidapi.com/news/{newspaperid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cyber-security-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def all_cyber_security_news(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This will show all Cyber Security News from around the world"
    
    """
    url = f"https://cyber-security-news.p.rapidapi.com/news"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "cyber-security-news.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

