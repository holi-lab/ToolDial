import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def mail_config_list(orderby: str='generatedAt|asc', limit: int=1000, filter: str='generatedAt>=2022-11-16T00:00:00.000Z', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get mail config list from cloud database"
    orderby: [generatedAt]|[asc, desc]
Only 'generatedAt' is allowed

        limit: Data get size limit

        filter: {field}{operator}{value}
[generatedAt][<=, >=, ==, !=, <, >]{value}
Only 'generatedAt' is allowed

        
    """
    url = f"https://xlsx-template.p.rapidapi.com/mail/config/list"
    querystring = {}
    if orderby:
        querystring['orderBy'] = orderby
    if limit:
        querystring['limit'] = limit
    if filter:
        querystring['filter'] = filter
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xlsx-template.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def generate_historys(orderby: str='serverTimingSec|asc', filter: str='serverTimingSec>=2', limit: int=10, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Generated files history"
    orderby: [serverTimingSec,serverTimingMs,generatedAt]|[asc, desc]
Only 'serverTimingSec', 'serverTimingMs', 'generatedAt' is allowed

        filter: {field}{operator}{value}
[serverTimingSec,serverTimingMs,generatedAt][<=, >=, ==, !=, <, >]{value}
Only 'serverTimingSec', 'serverTimingMs', 'generatedAt' is allowed

        limit: Data get size limit

        
    """
    url = f"https://xlsx-template.p.rapidapi.com/generate/historys"
    querystring = {}
    if orderby:
        querystring['orderBy'] = orderby
    if filter:
        querystring['filter'] = filter
    if limit:
        querystring['limit'] = limit
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xlsx-template.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mail_config_get(mailconfigid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get a mail confing from cloud database"
    
    """
    url = f"https://xlsx-template.p.rapidapi.com/mail/config/{mailconfigid}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xlsx-template.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def template_list(orderby: str='updated|asc', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Cloud storage templates file list"
    orderby: Only 'serverTimingSec', 'serverTimingMs', 'generatedAt' is allowed
[serverTimingSec,serverTimingMs,generatedAt]|[asc, desc]

        
    """
    url = f"https://xlsx-template.p.rapidapi.com/template/list"
    querystring = {}
    if orderby:
        querystring['orderBy'] = orderby
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xlsx-template.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def template_download(templatefilename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Download template file from Cloud storage"
    
    """
    url = f"https://xlsx-template.p.rapidapi.com/template/{templatefilename}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "xlsx-template.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

