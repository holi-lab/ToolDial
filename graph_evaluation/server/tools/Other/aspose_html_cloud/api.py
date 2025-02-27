import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def objectexists(path: str, storagename: str=None, versionid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File or folder path e.g. '/file.ext' or '/folder'
        storagename: Storage name
        versionid: File version ID
        
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/storage/exist/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    if versionid:
        querystring['versionId'] = versionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconvertdocumenttomhtmlbyurl(sourceurl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/convert/mhtml"
    querystring = {'sourceUrl': sourceurl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconvertdocumenttoxpsbyurl(sourceurl: str, leftmargin: int=None, bottommargin: int=None, rightmargin: int=None, height: int=None, width: int=None, topmargin: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/convert/xps"
    querystring = {'sourceUrl': sourceurl, }
    if leftmargin:
        querystring['leftMargin'] = leftmargin
    if bottommargin:
        querystring['bottomMargin'] = bottommargin
    if rightmargin:
        querystring['rightMargin'] = rightmargin
    if height:
        querystring['height'] = height
    if width:
        querystring['width'] = width
    if topmargin:
        querystring['topMargin'] = topmargin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentbyurl(sourceurl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/download"
    querystring = {'sourceUrl': sourceurl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def htmlverification(url: str, format: str='json', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/validator"
    querystring = {'url': url, }
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconvertdocumenttopdf(name: str, storage: str=None, folder: str=None, rightmargin: int=None, topmargin: int=None, bottommargin: int=None, leftmargin: int=None, height: int=None, width: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/{name}/convert/pdf"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if rightmargin:
        querystring['rightMargin'] = rightmargin
    if topmargin:
        querystring['topMargin'] = topmargin
    if bottommargin:
        querystring['bottomMargin'] = bottommargin
    if leftmargin:
        querystring['leftMargin'] = leftmargin
    if height:
        querystring['height'] = height
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
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
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/storage/{storagename}/exist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentfragmentsbycssselector(outformat: str, selector: str, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/{name}/fragments/css/{outformat}"
    querystring = {'selector': selector, }
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconvertdocumenttoimagebyurl(sourceurl: str, outformat: str, height: int=None, leftmargin: int=None, width: int=None, bottommargin: int=None, rightmargin: int=None, resolution: int=None, topmargin: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/convert/image/{outformat}"
    querystring = {'sourceUrl': sourceurl, }
    if height:
        querystring['height'] = height
    if leftmargin:
        querystring['leftMargin'] = leftmargin
    if width:
        querystring['width'] = width
    if bottommargin:
        querystring['bottomMargin'] = bottommargin
    if rightmargin:
        querystring['rightMargin'] = rightmargin
    if resolution:
        querystring['resolution'] = resolution
    if topmargin:
        querystring['topMargin'] = topmargin
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconvertdocumenttoxps(name: str, leftmargin: int=None, storage: str=None, folder: str=None, rightmargin: int=None, bottommargin: int=None, topmargin: int=None, height: int=None, width: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/{name}/convert/xps"
    querystring = {}
    if leftmargin:
        querystring['leftMargin'] = leftmargin
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    if rightmargin:
        querystring['rightMargin'] = rightmargin
    if bottommargin:
        querystring['bottomMargin'] = bottommargin
    if topmargin:
        querystring['topMargin'] = topmargin
    if height:
        querystring['height'] = height
    if width:
        querystring['width'] = width
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentfragmentsbycssselectorbyurl(selector: str, outformat: str, sourceurl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/fragments/css/{outformat}"
    querystring = {'selector': selector, 'sourceUrl': sourceurl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimportmarkdowntohtml(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/{name}/import/md"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconvertdocumenttopdfbyurl(sourceurl: str, bottommargin: int=None, rightmargin: int=None, topmargin: int=None, width: int=None, leftmargin: int=None, height: int=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/convert/pdf"
    querystring = {'sourceUrl': sourceurl, }
    if bottommargin:
        querystring['bottomMargin'] = bottommargin
    if rightmargin:
        querystring['rightMargin'] = rightmargin
    if topmargin:
        querystring['topMargin'] = topmargin
    if width:
        querystring['width'] = width
    if leftmargin:
        querystring['leftMargin'] = leftmargin
    if height:
        querystring['height'] = height
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def downloadfile(path: str, versionid: str=None, storagename: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File path e.g. '/folder/file.ext'
        versionid: File version ID to download
        storagename: Storage name
        
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/storage/file/{path}"
    querystring = {}
    if versionid:
        querystring['versionId'] = versionid
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentimagesbyurl(sourceurl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/images/all"
    querystring = {'sourceUrl': sourceurl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconvertdocumenttoimage(outformat: str, name: str, leftmargin: int=None, bottommargin: int=None, storage: str=None, topmargin: int=None, rightmargin: int=None, width: int=None, height: int=None, resolution: int=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/{name}/convert/image/{outformat}"
    querystring = {}
    if leftmargin:
        querystring['leftMargin'] = leftmargin
    if bottommargin:
        querystring['bottomMargin'] = bottommargin
    if storage:
        querystring['storage'] = storage
    if topmargin:
        querystring['topMargin'] = topmargin
    if rightmargin:
        querystring['rightMargin'] = rightmargin
    if width:
        querystring['width'] = width
    if height:
        querystring['height'] = height
    if resolution:
        querystring['resolution'] = resolution
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmergehtmltemplate(templatename: str, datapath: str, folder: str=None, options: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/{templatename}/merge"
    querystring = {'dataPath': datapath, }
    if folder:
        querystring['folder'] = folder
    if options:
        querystring['options'] = options
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getconvertdocumenttomarkdown(name: str, usegit: str='false', storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/{name}/convert/md"
    querystring = {}
    if usegit:
        querystring['useGit'] = usegit
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
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
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/storage/folder/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
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
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/storage/version/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentfragmentbyxpathbyurl(xpath: str, outformat: str, sourceurl: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/fragments/{outformat}"
    querystring = {'xPath': xpath, 'sourceUrl': sourceurl, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentimages(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/{name}/images/all"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentfragmentbyxpath(outformat: str, xpath: str, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/{name}/fragments/{outformat}"
    querystring = {'xPath': xpath, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
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
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/storage/disc"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get(addr: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    
    """
    url = f"https://aspose-html-cloud.p.rapidapi.com/html/seo"
    querystring = {'addr': addr, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-html-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

