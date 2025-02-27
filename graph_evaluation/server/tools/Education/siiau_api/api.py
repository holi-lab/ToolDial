import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_payment_order(x_student_code: str, x_student_nip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the payment order information of the student"
    x_student_code: The student code used to authenticate to the SIIAU system
        x_student_nip: The student nip used to authenticate to the SIIAU system
        
    """
    url = f"https://siiau-api.p.rapidapi.com/paymentorder"
    querystring = {'x-student-code': x_student_code, 'x-student-nip': x_student_nip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "siiau-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_schedule(x_student_code: str, x_student_nip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the schedule of the student from some calendar if specifed. If no calendar passed, then return the last calendar schedule"
    x_student_code: The student code used to authenticate to the SIIAU system
        x_student_nip: The student nip used to authenticate to the SIIAU system
        
    """
    url = f"https://siiau-api.p.rapidapi.com/schedule"
    querystring = {'x-student-code': x_student_code, 'x-student-nip': x_student_nip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "siiau-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_admission(x_student_code: str, x_student_nip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the admission information of the student"
    x_student_code: The student code used to authenticate to the SIIAU system
        x_student_nip: The student nip used to authenticate to the SIIAU system
        
    """
    url = f"https://siiau-api.p.rapidapi.com/admission"
    querystring = {'x-student-code': x_student_code, 'x-student-nip': x_student_nip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "siiau-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_credits_information(x_student_code: str, x_student_nip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the credits information about the student"
    x_student_code: The student code used to authenticate to the SIIAU system
        x_student_nip: The student nip used to authenticate to the SIIAU system
        
    """
    url = f"https://siiau-api.p.rapidapi.com/credits"
    querystring = {'x-student-code': x_student_code, 'x-student-nip': x_student_nip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "siiau-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_grades_kardex(x_student_code: str, x_student_nip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the grades per subject of the student from some calendar if specifed. Alternatively you can use the kardex endpoint, will produce the same output."
    x_student_code: The student code used to authenticate to the SIIAU system
        x_student_nip: The student nip used to authenticate to the SIIAU system
        
    """
    url = f"https://siiau-api.p.rapidapi.com/grades"
    querystring = {'x-student-code': x_student_code, 'x-student-nip': x_student_nip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "siiau-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def login_status(x_student_code: str, x_student_nip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Check if the credentials are correct"
    x_student_code: The student code used to authenticate to the SIIAU system
        x_student_nip: The student nip used to authenticate to the SIIAU system
        
    """
    url = f"https://siiau-api.p.rapidapi.com/student/login"
    querystring = {'x-student-code': x_student_code, 'x-student-nip': x_student_nip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "siiau-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_student_progress(x_student_code: str, x_student_nip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the progress of the student by Semester-Calendar/Career/Campus"
    x_student_code: The student code used to authenticate to the SIIAU system
        x_student_nip: The student nip used to authenticate to the SIIAU system
        
    """
    url = f"https://siiau-api.p.rapidapi.com/student/progress"
    querystring = {'x-student-code': x_student_code, 'x-student-nip': x_student_nip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "siiau-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_student_information(x_student_code: str, x_student_nip: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Retrieves the general information about the student"
    x_student_code: The student code used to authenticate to the SIIAU system
        x_student_nip: The student nip used to authenticate to the SIIAU system
        
    """
    url = f"https://siiau-api.p.rapidapi.com/student"
    querystring = {'x-student-code': x_student_code, 'x-student-nip': x_student_nip, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "siiau-api.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

