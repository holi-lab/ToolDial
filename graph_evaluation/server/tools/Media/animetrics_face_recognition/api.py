import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def assign_face_to_subject(subject_id: str, face_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Assign Face ID to a Subject ID. This operation will change the Subject ID associated with an individual Face ID. The Face ID that is required should be taken from the response of a previous enrollment. Assigning a Face ID to a new Subject ID who belongs to more than one gallery will affect that Subject's identity in each gallery. This operation might be used if several pictures of an individual are enrolled and then later it is found out that one of these pictures was in fact someone else."
    subject_id: unique subject ID to replace that of a previous enrollment
        face_id: face ID from the response of a previous enrollment
        
    """
    url = f"https://animetrics.p.rapidapi.com/assign_face_to_subject"
    querystring = {'subject_id': subject_id, 'face_id': face_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animetrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_remove_from_gallery(subject_id: str, gallery_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Removes an already enrolled Subject from an existing Gallery. If this is the only subject in that Gallery, the Gallery will be deleted. If the Subject exists in no other Galleries, he will be deleted from the Face Recognition System."
    subject_id: The ID used to previously enroll some person ("subject")
        gallery_id: A unique ID indicating the collection from which the subject should be removed. If the subject exists in no other galleries, the biometric template will be removed from the Facial Recognition System and subsequent Add To Gallery operations for this subject will fail.
        
    """
    url = f"https://animetrics.p.rapidapi.com/v2/remove_from_gallery"
    querystring = {'subject_id': subject_id, 'gallery_id': gallery_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animetrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_usage(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieve your current daily and monthly count of function calls made to the api. Total, billable, detect, enroll and recognize counts are displayed."
    
    """
    url = f"https://animetrics.p.rapidapi.com/v2/usage"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animetrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_view_gallery(gallery_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View the subject ids that have been enrolled in a specific gallery."
    gallery_id: A unique ID indicating the gallery to query.
        
    """
    url = f"https://animetrics.p.rapidapi.com/v2/view_gallery"
    querystring = {'gallery_id': gallery_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animetrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_view_subject(subject_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View the face ids that have been enrolled in a specific subject."
    subject_id: unique subject ID that matches one or more previous enrollments
        
    """
    url = f"https://animetrics.p.rapidapi.com/v2/view_subject"
    querystring = {'subject_id': subject_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animetrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_add_to_gallery(subject_id: str, gallery_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Adds an already enrolled Subject into an existing or uncreated Gallery. Galleries that don't exist will automatically be created."
    subject_id: The ID used to previously enroll some person ("subject")
        gallery_id: A unique ID indicating the collection to which the subject should be added. If this gallery doesn't exist, it will be created.
        
    """
    url = f"https://animetrics.p.rapidapi.com/v2/add_to_gallery"
    querystring = {'subject_id': subject_id, 'gallery_id': gallery_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animetrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_assign_face_to_subject(face_id: str, subject_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Assign Face ID to a Subject ID. This operation will change the Subject ID associated with an individual Face ID. The Face ID that is required should be taken from the response of a previous enrollment. Assigning a Face ID to a new Subject ID who belongs to more than one gallery will affect that Subject's identity in each gallery. This operation might be used if several pictures of an individual are enrolled and then later it is found out that one of these pictures was in fact someone else."
    face_id: face ID from the response of a previous enrollment
        subject_id: unique subject ID to replace that of a previous enrollment
        
    """
    url = f"https://animetrics.p.rapidapi.com/v2/assign_face_to_subject"
    querystring = {'face_id': face_id, 'subject_id': subject_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animetrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def verify(subject_id_of_target: str, subject_id_of_unknown: str='John_Doe', toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Verify if an unknown face (or collection of faces of an unknown individual) has the same identity as a claimed target. A similarity score is returned, and is optimized for discriminating the target subject from other individuals. Note that there must be at least two faces of the target subject enrolled."
    subject_id_of_target: A unique subject ID of the known person (must have at least 2 faces enrolled)
        subject_id_of_unknown: A unique subject ID of the person who's identity is to be verified
        
    """
    url = f"https://animetrics.p.rapidapi.com/verify"
    querystring = {'subject_id_of_target': subject_id_of_target, }
    if subject_id_of_unknown:
        querystring['subject_id_of_unknown'] = subject_id_of_unknown
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animetrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def view_subject(subject_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "View the face ids that have been enrolled in a specific subject."
    subject_id: unique subject ID that matches one or more previous enrollments
        
    """
    url = f"https://animetrics.p.rapidapi.com/view_subject"
    querystring = {'subject_id': subject_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animetrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def delete_face(face_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Delete a Face from a known Subject. This operation will only delete the individual Face that is associated with an enrollment transaction for a Subject. The Face ID that is required should be taken from the response of a previous enrollment. Deleteing a Face from a Subject who belongs to more than one gallery will affect that Subject's identity in each gallery. This operation might be used if several pictures of an individual are enrolled and then later it is found out that one of these pictures was in fact someone else who's identity is unknown."
    face_id: face ID from the response of a previous enrollment
        
    """
    url = f"https://animetrics.p.rapidapi.com/delete_face"
    querystring = {'face_id': face_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animetrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def v2_delete_face(face_id: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Delete a Face from a known Subject. This operation will only delete the individual Face that is associated with an enrollment transaction for a Subject. The Face ID that is required should be taken from the response of a previous enrollment. Deleteing a Face from a Subject who belongs to more than one gallery will affect that Subject's identity in each gallery. This operation might be used if several pictures of an individual are enrolled and then later it is found out that one of these pictures was in fact someone else who's identity is unknown."
    face_id: Delete a Face from a known Subject. This operation will only delete the individual Face that is associated with an enrollment transaction for a Subject. The Face ID that is required should be taken from the response of a previous enrollment. Deleteing a Face from a Subject who belongs to more than one gallery will affect that Subject's identity in each gallery. This operation might be used if several pictures of an individual are enrolled and then later it is found out that one of these pictures was in fact someone else who's identity is unknown.
        
    """
    url = f"https://animetrics.p.rapidapi.com/v2/delete_face"
    querystring = {'face_id': face_id, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "animetrics.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

