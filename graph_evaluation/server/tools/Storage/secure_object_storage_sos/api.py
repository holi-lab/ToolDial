import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getbucketinfo(bucket: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves general information on a bucket, authorizing via API-Key"
    bucket: bucket name
        
    """
    url = f"https://secure-object-storage-sos.p.rapidapi.com/v1/x/{bucket}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "secure-object-storage-sos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getrecbindirlist(maxmd: str, sf: str, fo: str, bucket: str, maxcd: str, minmd: str, sa: str, mincd: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the list of objects in the bucket's recycle bin, authorizing via API-Key"
    maxmd: maximum modification date, in the format 2006-01-02T15:04:05
        sf: sort filter: 'name'||'objType'||'createdBy'||'creationDate'||'fileDate'||'objSize'
        f: name of field for regex query: 'name'||'objType'||'createdBy'||'creationDate'||'fileDate'||'objSize'
        fo: if f is present, fo contains the regex to be matched
        bucket: bucket name
        maxcd: maximum creation date, in the format 2006-01-02T15:04:05
        minmd: minimum modification date, in the format 2006-01-02T15:04:05
        sa: to be used together with sf, 'true' if sort ascending, 'false' if sort descending
        mincd: minimum creation date, in the format 2006-01-02T15:04:05
        
    """
    url = f"https://secure-object-storage-sos.p.rapidapi.com/v1/r/{bucket}"
    querystring = {'maxmd': maxmd, 'sf': sf, 'fo': fo, 'maxcd': maxcd, 'minmd': minmd, 'sa': sa, 'mincd': mincd, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "secure-object-storage-sos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdirlist(objpath: str, sf: str, fo: str, maxmd: str, mincd: str, bucket: str, minmd: str, sa: str, maxcd: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the list of a directory inside a bucket, authorizing via API-Key"
    objpath: full directory path
        sf: sort filter: 'name'||'objType'||'createdBy'||'creationDate'||'fileDate'||'objSize'
        fo: if f is present, fo contains the regex to be matched
        f: name of field for regex query: 'name'||'objType'||'createdBy'||'creationDate'||'fileDate'||'objSize'
        maxmd: maximum modification date, in the format 2006-01-02T15:04:05
        mincd: minimum creation date, in the format 2006-01-02T15:04:05
        bucket: bucket name
        minmd: minimum modification date, in the format 2006-01-02T15:04:05
        sa: to be used together with sf, 'true' if sort ascending, 'false' if sort descending
        maxcd: maximum creation date, in the format 2006-01-02T15:04:05
        
    """
    url = f"https://secure-object-storage-sos.p.rapidapi.com/v1/d/{bucket}/{objpath}"
    querystring = {'sf': sf, 'fo': fo, 'maxmd': maxmd, 'mincd': mincd, 'minmd': minmd, 'sa': sa, 'maxcd': maxcd, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "secure-object-storage-sos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getobjectinfo(bucket: str, objpath: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves object Information/Metadata, authorizing via API-Key"
    bucket: bucket name
        objpath: full object path
        
    """
    url = f"https://secure-object-storage-sos.p.rapidapi.com/v1/i/{bucket}/{objpath}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "secure-object-storage-sos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadfile(bucket: str, objpath: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Reassembles, decrypts, and downloads a file from a bucket, authorizing via API-Key"
    bucket: bucket name
        objpath: full object path
        
    """
    url = f"https://secure-object-storage-sos.p.rapidapi.com/v1/f/{bucket}/{objpath}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "secure-object-storage-sos.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

