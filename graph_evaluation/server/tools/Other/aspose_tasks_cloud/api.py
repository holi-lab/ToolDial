import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcalendar(name: str, calendaruid: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        calendaruid: Calendar's Uid
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/calendars/{calendaruid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcalendars(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/calendars"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getoutlinecodes(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/outlineCodes"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcriticalpath(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/criticalPath"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettaskrecurringinfo(taskuid: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    taskuid: Task Uid
        name: The name of the file.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/tasks/{taskuid}/recurringInfo"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getresourcetimephaseddata(name: str, resourceuid: int, enddate: str=None, folder: str=None, type: str='AssignmentRemainingWork', startdate: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        resourceuid: Uid of resource to get timephased data for.
        enddate: End date.
        folder: The document folder.
        type: Type of timephased data to get.
        startdate: Start date.
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/resources/{resourceuid}/timeScaleData"
    querystring = {}
    if enddate:
        querystring['endDate'] = enddate
    if folder:
        querystring['folder'] = folder
    if type:
        querystring['type'] = type
    if startdate:
        querystring['startDate'] = startdate
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getprojectids(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/projectids"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
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
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/storage/folder/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getresource(resourceuid: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    resourceuid: Resource Uid
        name: The name of the file.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/resources/{resourceuid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcalendarexceptions(calendaruid: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    calendaruid: Calendar's Uid
        name: The name of the file.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/calendars/{calendaruid}/calendarExceptions"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettaskdocumentwithformat(format: str, name: str, returnasziparchive: bool=None, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    format: The format of the resulting file.
        name: The name of the file.
        returnasziparchive: If parameter is true, HTML resources are included as separate files and returned along with the resulting html file as a zip package.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/format"
    querystring = {'format': format, }
    if returnasziparchive:
        querystring['returnAsZipArchive'] = returnasziparchive
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettasktimephaseddata(name: str, taskuid: int, enddate: str=None, startdate: str=None, folder: str=None, type: str='AssignmentRemainingWork', storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        taskuid: Uid of task to get timephased data for.
        enddate: End date.
        startdate: Start date.
        folder: The document folder.
        type: Type of timephased data to get.
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/tasks/{taskuid}/timeScaleData"
    querystring = {}
    if enddate:
        querystring['endDate'] = enddate
    if startdate:
        querystring['startDate'] = startdate
    if folder:
        querystring['folder'] = folder
    if type:
        querystring['type'] = type
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getprojectlist(siteurl: str, username: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    siteurl: The URL of PWA (Project Web Access) API of Project Online.
        username: The user name for the sharepoint site.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/projectOnline/projectList"
    querystring = {'siteUrl': siteurl, }
    if username:
        querystring['userName'] = username
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettaskassignments(name: str, taskuid: int, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        taskuid: Task Uid
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/tasks/{taskuid}/assignments"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettasklinks(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/taskLinks"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getriskanalysisreport(taskuid: int, name: str, iterations: int=100, pessimistic: int=130, optimistic: int=70, folder: str=None, filename: str=None, distributiontype: str='1', storage: str=None, confidencelevel: str='75', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    taskuid: The uid of the task for which this risk will be applied in Monte Carlo simulation.
        name: The name of the file.
        iterations: The number of iterations to use in Monte Carlo simulation. The default value is 100.
        pessimistic: The percentage of the most likely task duration which can happen in the worst possible project scenario. The default value is 125, which means that if the estimated specified task duration is 4 days then the pessimistic duration will be 5 days.
        optimistic: The percentage of the most likely task duration which can happen in the best possible project scenario. The default value is 75, which means that if the estimated specified task duration is 4 days then the optimistic duration will be 3 days.
        folder: The folder storage.
        filename: The resulting report fileName.
        distributiontype: Gets or sets the probability distribution used in Monte Carlo simulation. The default value is ProbabilityDistributionType.Normal.
        storage: The document storage.
        confidencelevel: Gets or sets the confidence level that correspond to the percentage of the time the actual generated values will be within optimistic and pessimistic estimates. The default value is CL99.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/riskAnalysisReport"
    querystring = {'taskUid': taskuid, }
    if iterations:
        querystring['iterations'] = iterations
    if pessimistic:
        querystring['pessimistic'] = pessimistic
    if optimistic:
        querystring['optimistic'] = optimistic
    if folder:
        querystring['folder'] = folder
    if filename:
        querystring['fileName'] = filename
    if distributiontype:
        querystring['distributionType'] = distributiontype
    if storage:
        querystring['storage'] = storage
    if confidencelevel:
        querystring['confidenceLevel'] = confidencelevel
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
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
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/storage/version/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getcalendarworkweeks(calendaruid: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    calendaruid: Calendar Uid
        name: The name of the file.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/calendars/{calendaruid}/workWeeks"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getextendedattributes(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/extendedAttributes"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
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
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/storage/disc"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettasks(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/tasks"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
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
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/storage/{storagename}/exist"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getreportpdf(name: str, type: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        type: A type of the project's graphical report.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/report"
    querystring = {'type': type, }
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getvbaproject(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file
        folder: The folder storage
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/vbaproject"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getassignment(name: str, assignmentuid: int, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        assignmentuid: Assignment Uid
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/assignments/{assignmentuid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getresourceassignments(resourceuid: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    resourceuid: Resource Uid
        name: The name of the file.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/resources/{resourceuid}/assignments"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettask(taskuid: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    taskuid: Task Uid
        name: The name of the file.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/tasks/{taskuid}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
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
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/storage/file/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    if versionid:
        querystring['versionId'] = versionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentproperty(propertyname: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    propertyname: The property name.
        name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/documentproperties/{propertyname}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getextendedattributebyindex(index: int, name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    index: Index (See ExtendedAttributeItem.Index property) or FieldId of the extended attribute.
        name: The name of the file.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/extendedAttributes/{index}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getassignments(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/assignments"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getdocumentproperties(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The document name.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/documentproperties"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getwbsdefinition(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/wbsDefinition"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getresources(name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/resources"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getassignmenttimephaseddata(name: str, assignmentuid: int, type: str='AssignmentRemainingWork', storage: str=None, enddate: str=None, startdate: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        assignmentuid: Uid of assignment to get timephased data for.
        type: Type of timephased data to get.
        storage: The document storage.
        enddate: End date.
        startdate: Start date.
        folder: The document folder.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/assignments/{assignmentuid}/timeScaleData"
    querystring = {}
    if type:
        querystring['type'] = type
    if storage:
        querystring['storage'] = storage
    if enddate:
        querystring['endDate'] = enddate
    if startdate:
        querystring['startDate'] = startdate
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def objectexists(path: str, storagename: str=None, versionid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    path: File or folder path e.g. '/file.ext' or '/folder'
        storagename: Storage name
        versionid: File version ID
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/storage/exist/{path}"
    querystring = {}
    if storagename:
        querystring['storageName'] = storagename
    if versionid:
        querystring['versionId'] = versionid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpagecount(name: str, storage: str=None, enddate: str=None, presentationformat: str='GanttChart', timescale: str='1', folder: str=None, pagesize: str='Letter', startdate: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        storage: The document storage.
        enddate: End date to get page count for.
        presentationformat: PresentationFormat to get page count for.
        timescale: Timescale to get page count for.
        folder: The document folder
        pagesize: PageSize to get page count for.
        startdate: Start date to get page count for.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/pagecount"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if enddate:
        querystring['endDate'] = enddate
    if presentationformat:
        querystring['presentationFormat'] = presentationformat
    if timescale:
        querystring['timescale'] = timescale
    if folder:
        querystring['folder'] = folder
    if pagesize:
        querystring['pageSize'] = pagesize
    if startdate:
        querystring['startDate'] = startdate
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getoutlinecodebyindex(index: int, name: str, storage: str=None, folder: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    index: Index of the outline code. See OutlineCodeItem.Index property.
        name: The name of the file.
        storage: The document storage.
        folder: The document folder.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}/outlineCodes/{index}"
    querystring = {}
    if storage:
        querystring['storage'] = storage
    if folder:
        querystring['folder'] = folder
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def gettaskdocument(name: str, folder: str=None, storage: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    " "
    name: The name of the file.
        folder: The document folder.
        storage: The document storage.
        
    """
    url = f"https://aspose-tasks-cloud.p.rapidapi.com/tasks/{name}"
    querystring = {}
    if folder:
        querystring['folder'] = folder
    if storage:
        querystring['storage'] = storage
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "aspose-tasks-cloud.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

