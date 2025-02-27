import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def documentdownload(folderid: int, docnumber: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can download an individual document as a pdf file.
		"
    folderid: Numeric ID of the folder to download.
        docnumber: The index of the document starting from 1 whose PDF you want to download.
        
    """
    url = f"https://foxit-esign.p.rapidapi.com/folders/document/download"
    querystring = {'folderId': folderid, 'docNumber': docnumber, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foxit-esign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def activityhistory(folderid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get an ordered list of activities that occurred on a specific folder.
		"
    folderid: Numeric ID of the folder to get
        
    """
    url = f"https://foxit-esign.p.rapidapi.com/folders/viewActivityHistory"
    querystring = {'folderId': folderid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foxit-esign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def myfolder(folderid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can poll our API to get the folder/document data from any folder that you have created.
		"
    folderid: ID of the folder to get
        
    """
    url = f"https://foxit-esign.p.rapidapi.com/folders/myfolder"
    querystring = {'folderId': folderid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foxit-esign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def deletedlog(folderid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the history of a folder which has been deleted
		"
    folderid: Numeric ID of the folder to get
        
    """
    url = f"https://foxit-esign.p.rapidapi.com/folders/deletedLog"
    querystring = {'folderId': folderid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foxit-esign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def folderdownload(folderid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can poll our API to get the folder/document data from any folder that you have created.
		"
    folderid: Numeric ID of the folder to get
        
    """
    url = f"https://foxit-esign.p.rapidapi.com/folders/download"
    querystring = {'folderId': folderid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foxit-esign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def templateslist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "You can get a list of all the templates from your Foxit eSign into your application using the following call
		"
    
    """
    url = f"https://foxit-esign.p.rapidapi.com/templates/list"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foxit-esign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mytemplate(templateid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get all information with regards to a specific template
		"
    templateid: Numeric ID of the template to receive.
        
    """
    url = f"https://foxit-esign.p.rapidapi.com/folders/mytemplate"
    querystring = {'templateId': templateid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foxit-esign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def mychannel(channelid: int, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive all information regarding a specific webhook channel.
		"
    channelid: Numeric ID of the channel to receive details about.
        
    """
    url = f"https://foxit-esign.p.rapidapi.com/webhook/mychannel"
    querystring = {'channelId': channelid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foxit-esign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def channellist(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Receive a list of webhook channel details in your account.
		"
    
    """
    url = f"https://foxit-esign.p.rapidapi.com/webhook/channellist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "foxit-esign.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

