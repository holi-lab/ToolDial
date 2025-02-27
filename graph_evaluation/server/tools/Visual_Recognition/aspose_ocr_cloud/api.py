import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def storageexists(storagename: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    storagename: Storage name
        
    """
    url = f"https://aspose-ocr-cloud.p.rapidapi.com/ocr/storage/{storagename}/exist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-ocr-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hcpassthrough(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-ocr-cloud.p.rapidapi.com/ocr/hc_full"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-ocr-cloud.p.rapidapi.com"
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
    url = f"https://aspose-ocr-cloud.p.rapidapi.com/ocr/storage/version/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-ocr-cloud.p.rapidapi.com"
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
    url = f"https://aspose-ocr-cloud.p.rapidapi.com/ocr/storage/folder/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-ocr-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrecognizedocument(name: str, folder: str=None, resulttype: int=1, skewcorrect: bool=None, storage: str=None, language: int=1, dsrmode: int=2, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: Name of the file to recognize.
        folder: The image folder.
        resulttype: The type of result: Text, HOCR, PDF or their combinations
        skewcorrect: An option to switch skew correction algorithm
        storage: The image storage.
        language: Language for OCR Core Aspose.OCR.Core.Transport 
                    BaseStructures.Language.LanguageGroup
                
            
        dsrmode: An option to switch DSR algorithm
        
    """
    url = f"https://aspose-ocr-cloud.p.rapidapi.com/ocr/{name}/recognize"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if resulttype:
        querystring['resultType'] = resulttype
    if skewcorrect:
        querystring['skewCorrect'] = skewcorrect
    if storage:
        querystring['storage'] = storage
    if language:
        querystring['language'] = language
    if dsrmode:
        querystring['dsrMode'] = dsrmode
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-ocr-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hclocal(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-ocr-cloud.p.rapidapi.com/ocr/hc_wrapper"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-ocr-cloud.p.rapidapi.com"
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
    url = f"https://aspose-ocr-cloud.p.rapidapi.com/ocr/storage/disc"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-ocr-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadfile(path: str, storagename: str=None, versionid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File path e.g. '/folder/file.ext'
        storagename: Storage name
        versionid: File version ID to download
        
    """
    url = f"https://aspose-ocr-cloud.p.rapidapi.com/ocr/storage/file/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    if versionid:
        querystring['versionId'] = versionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-ocr-cloud.p.rapidapi.com"
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
    url = f"https://aspose-ocr-cloud.p.rapidapi.com/ocr/storage/exist/{path}"
    querystring = {}
    if versionid:
        querystring['versionId'] = versionid
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-ocr-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

