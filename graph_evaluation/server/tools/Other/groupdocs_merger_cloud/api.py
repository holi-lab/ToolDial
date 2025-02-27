import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def downloadfile(path: str, storagename: str=None, versionid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File path e.g. '/folder/file.ext'
        storagename: Storage name
        versionid: File version ID to download
        
    """
    url = f"https://groupdocs-merger-cloud1.p.rapidapi.com/merger/storage/file/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    if versionid:
        querystring['versionId'] = versionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "groupdocs-merger-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfileversions(path: str, storagename: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File path e.g. '/file.ext'
        storagename: Storage name
        
    """
    url = f"https://groupdocs-merger-cloud1.p.rapidapi.com/merger/storage/version/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "groupdocs-merger-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getsupportedfileformats(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://groupdocs-merger-cloud1.p.rapidapi.com/merger/formats"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "groupdocs-merger-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdiscusage(storagename: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    storagename: Storage name
        
    """
    url = f"https://groupdocs-merger-cloud1.p.rapidapi.com/merger/storage/disc"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "groupdocs-merger-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def objectexists(path: str, versionid: str=None, storagename: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File or folder path e.g. '/file.ext' or '/folder'
        versionid: File version ID
        storagename: Storage name
        
    """
    url = f"https://groupdocs-merger-cloud1.p.rapidapi.com/merger/storage/exist/{path}"
    querystring = {}
    if versionid:
        querystring['versionId'] = versionid
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "groupdocs-merger-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def storageexists(storagename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    storagename: Storage name
        
    """
    url = f"https://groupdocs-merger-cloud1.p.rapidapi.com/merger/storage/{storagename}/exist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "groupdocs-merger-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getfileslist(path: str, storagename: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: Folder path e.g. '/folder'
        storagename: Storage name
        
    """
    url = f"https://groupdocs-merger-cloud1.p.rapidapi.com/merger/storage/folder/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "groupdocs-merger-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def checkpassword(filepath: str, storagename: str=None, versionid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    filepath: File path
        storagename: Storage name
        versionid: Version Id
        
    """
    url = f"https://groupdocs-merger-cloud1.p.rapidapi.com/merger/password"
    querystring = {'filePath': filepath, }
    if storagename:
        querystring['storageName'] = storagename
    if versionid:
        querystring['versionId'] = versionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "groupdocs-merger-cloud1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

