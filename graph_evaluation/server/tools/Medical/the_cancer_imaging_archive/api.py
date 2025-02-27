import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def getcollectionvalues(format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Set of all TCIA collection names"
    format: Specify output type. Allowed values CSV/HTML/XML/JSON
        
    """
    url = f"https://tcia.p.rapidapi.com/getCollectionValues"
    querystring = {}
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tcia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmodalityvalues(collection: str=None, bodypartexamined: str=None, modality: str=None, format: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "getModalityValues"
    collection: A label used to name a set of images collected for a specific trial or other reason.  Assigned during the process of curating the data.
        bodypartexamined: Entered on a per collection basis using relevant SNOMED terms. (DICOM Tag  0018x0015)
        modality: Standard DICOM definition (DICOM Tag 0008x0060)
        format: Specify output type. Allowed values CSV/HTML/XML/JSON
        
    """
    url = f"https://tcia.p.rapidapi.com/getModalityValues"
    querystring = {}
    if collection:
        querystring['Collection'] = collection
    if bodypartexamined:
        querystring['BodyPartExamined'] = bodypartexamined
    if modality:
        querystring['Modality'] = modality
    if format:
        querystring['format'] = format
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tcia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getbodypartvalues(format: str=None, collection: str=None, bodypartexamined: str=None, modality: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Set of all body part names filtered by query keys"
    format: Specify output type. Allowed values CSV/HTML/XML/JSON
        collection: A label used to name a set of images collected for a specific trial or other reason.  Assigned during the process of curating the data.
        bodypartexamined: Standard DICOM definition (0018x0015)
        modality: Standard DICOM definition (0008x0060)
        
    """
    url = f"https://tcia.p.rapidapi.com/getBodyPartValues"
    querystring = {}
    if format:
        querystring['format'] = format
    if collection:
        querystring['Collection'] = collection
    if bodypartexamined:
        querystring['BodyPartExamined'] = bodypartexamined
    if modality:
        querystring['Modality'] = modality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tcia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpatientstudy(format: str=None, collection: str=None, patientid: str=None, studyinstanceuid: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a set of patient/study objects filtered by query keys"
    format: Specify output type. Allowed values CSV/HTML/XML/JSON
        collection: A label used to name a set of images collected for a specific trial or other reason.  Assigned during the process of curating the data.
        patientid: Has been de-identified as part of submission process (DICOM Tag  0010x0020)
        studyinstanceuid: Has been de-identified as part of submission process. (DICOM Tag 0020x000D)
        
    """
    url = f"https://tcia.p.rapidapi.com/getPatientStudy"
    querystring = {}
    if format:
        querystring['format'] = format
    if collection:
        querystring['Collection'] = collection
    if patientid:
        querystring['PatientID'] = patientid
    if studyinstanceuid:
        querystring['StudyInstanceUID'] = studyinstanceuid
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tcia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getseries(format: str=None, collection: str=None, patientid: str=None, studyinstanceuid: str=None, modality: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a set of series objects filtered by query keys"
    format: Specify output type. Allowed values CSV/HTML/XML/JSON
        collection: A label used to name a set of images collected for a specific trial or other reason.  Assigned during the process of curating the data.
        patientid: Has been de-identified as part of submission process (DICOM Tag  0010x0020)
        studyinstanceuid: Has been de-identified as part of submission process. (DICOM Tag 0020x000D)
        modality: Standard DICOM definition (0008x0060)
        
    """
    url = f"https://tcia.p.rapidapi.com/getSeries"
    querystring = {}
    if format:
        querystring['format'] = format
    if collection:
        querystring['Collection'] = collection
    if patientid:
        querystring['PatientID'] = patientid
    if studyinstanceuid:
        querystring['StudyInstanceUID'] = studyinstanceuid
    if modality:
        querystring['Modality'] = modality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tcia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getimage(seriesinstanceuid: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Set of images in a zip file. NOTE: The resulting zip files can become very large. We strongly advise not testing  this endpoint via Mashape. Instead, consider using curl with the -o option set to output.zip"
    seriesinstanceuid: Has been de-identified as part of submission process. (DICOM Tag 0020x000E)
        
    """
    url = f"https://tcia.p.rapidapi.com/getImage"
    querystring = {'SeriesInstanceUID': seriesinstanceuid, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tcia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getpatient(format: str=None, collection: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a set of patient objects filtered by query keys"
    format: Specify output type. Allowed values CSV/HTML/XML/JSON
        collection: A label used to name a set of images collected for a specific trial or other reason.  Assigned during the process of curating the data.
        
    """
    url = f"https://tcia.p.rapidapi.com/getPatient"
    querystring = {}
    if format:
        querystring['format'] = format
    if collection:
        querystring['Collection'] = collection
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tcia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def getmanufacturervalues(format: str=None, collection: str=None, bodypartexamined: str=None, modality: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Returns a set of all manufacturer names filtered by query keys"
    format: Specify output type. Allowed values CSV/HTML/XML/JSON
        collection: A label used to name a set of images collected for a specific trial or other reason.  Assigned during the process of curating the data.
        bodypartexamined: Standard DICOM definition (0018x0015)
        modality: Standard DICOM definition (0008x0060)
        
    """
    url = f"https://tcia.p.rapidapi.com/getManufacturerValues"
    querystring = {}
    if format:
        querystring['format'] = format
    if collection:
        querystring['Collection'] = collection
    if bodypartexamined:
        querystring['BodyPartExamined'] = bodypartexamined
    if modality:
        querystring['Modality'] = modality
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "tcia.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

