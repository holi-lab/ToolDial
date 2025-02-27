import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def upload(p: str, path: str, pc: str, u: str, content: str='/Documents/myfile.txt', content64: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "With this API you can upload a file to your online account using a simple rest call. If you don't have an account yet go on www.memopal.com, download the software and create one.  Or create a free 3GB account here:
		https://apps.memopal.com/l/activate-trial.php"
    p: mypassword
        pc: Memopal is organized in not less then three layers of subfolders:  Computer/first_folders/other_folders_or_files so you need to specify always at leat a computer name and a folder
        u: email address for accessing to your account
        content: at least one folder is required then the name of the file
        content64: base64 encoded the content of the file
        
    """
    url = f"https://memopal-cloud-storage-put-file.p.rapidapi.com/api/file/upload"
    querystring = {'p': p, 'path': path, 'pc': pc, 'u': u, }
    if content:
        querystring['content'] = content
    if content64:
        querystring['content64'] = content64
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "memopal-cloud-storage-put-file.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

