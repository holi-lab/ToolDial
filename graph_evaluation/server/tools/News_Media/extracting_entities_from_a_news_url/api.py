import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def https_spacyner_pregnya_com_getentitiesfromurl_url_https_www_nytimes_com_2020_06_18_us_politics_john_bolton_memoir_takeaways_html(url: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "spacyner.pregnya.com
		The "getEntitiesfromURL" API endpoint presented by ner.pregnya.com gives the list of entities from the given URL of a NEWS web page. This API works well with a NEWS web page. The entities are of PEOPLE, PLACES, THINGS, EVENTS, ORGANIZATION, CORDINAS, DATE, TIME, MONEY, and WORK_OF_ART.
		
		This "getEntitiesfromURL" gives the following information:
		1). Total_number_of_Entities
		2). The TOP 5 most common Entities
		3). Entities Group Aggregates. Aggregate total of each entities.
		4). Total number of distinct Entities
		5). Distinct Entities with Types"
    
    """
    url = f"https://extracting-entities-from-a-news-url.p.rapidapi.com/getEntitiesfromURL"
    querystring = {'url': url, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "extracting-entities-from-a-news-url.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

