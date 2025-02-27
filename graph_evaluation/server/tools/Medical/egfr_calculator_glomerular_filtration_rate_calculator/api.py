import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def egfr_calculation_for_creatinine_value_in_mol_l(female: bool, creatmicromol: int, age: int=50, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API documentation for eGFR2 (for Creatinine value in μmol/L)
		
		
		
		
		Introduction
		
		
		This API provides a way to calculate the Estimated Glomerular Filtration Rate (eGFR) for assessing renal function using the 2021 CKD-EPI formula. The eGFR is a measure of the kidney's ability to filter waste from the blood, and is an important indicator of renal health.
		Endpoint
		
		The API endpoint for the eGFR calculation is:
		
		
		https://api.algomed.in/egfr2
		
		
		Input
		
		The API requires the following input parameters to be passed in the request body as a JSON object:
		* creatinine (float): The patient's creatinine level, in μmol/L. 
		* age (integer): The patient's age, in years.
		* Female (boolean): Is the patient a female, True or false.
		
		
		Input example:
		
		Here is an example for an input request 
		
		https://api.algomed.in/egfr1?creat=100.0&female=true&age=50
		
		Here the creatinine  is 100.0 μmol/L, The gender is female (Female = true) and age of the patient is 50 years. 
		
		curl -X 'GET' \
		  'https://api.algomed.in/egfr1?creat=1.2&female=true&age=50' \
		  -H 'accept: application/json'
		
		
		Output
		The API returns a JSON object with the following structure:
		
		{
		  "egfr": 59, ==>  The calculated eGFR value
		  "units": "mL/min/1.73m2",  ==> The units for the same
		  "CKD category": "G3a", ==> CKD category based on current guidelines 
		  "details": "This falls in category of mildly to moderately decreased eGFR" ==> Information about the CKD category
		}
		
		
		
		Example
		Here is an example of a request to the API endpoint:
		
		GET https://api.algomed.in/egfr1?creat=100.0&female=true&age=50
		
		And the corresponding response from the API:
		
		{
		  "egfr": 59,
		  "units": "mL/min/1.73m2",
		  "CKD category": "G3a",
		  "details": "This falls in category of mildly to moderately decreased eGFR"
		}
		
		
		Error Codes
		In case of any errors, the API returns a JSON object with the following structure:
		* error: A string indicating the error that occurred.
		* message: A detailed error message.
		Notes
		* The creatinine value must be entered in μmol/L.
		* The API only accepts requests with the 'application/json' content type.
		* The API returns a 500 Internal Server Error response in case of any internal errors.
		Disclaimer
		This API is provided for informational purposes only and is not intended to be used for medical advice, diagnosis, or treatment. The results of the eGFR calculation should always be interpreted in the context of a comprehensive medical evaluation by a qualified healthcare professional."
    
    """
    url = f"https://egfr-calculator-glomerular-filtration-rate-calculator.p.rapidapi.com/egfr2"
    querystring = {'female': female, 'creatmicromol': creatmicromol, }
    if age:
        querystring['age'] = age
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "egfr-calculator-glomerular-filtration-rate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def egfr_calculation_for_creatinine_value_in_mg_dl(age: int, creat: int, female: bool, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "API documentation for eGFR1 (for Creatinine value in mg/dl)
		
		
		API Documentation: eGFR Calculator API
		
		Introduction
		
		
		This API provides a way to calculate the Estimated Glomerular Filtration Rate (eGFR) for assessing renal function using the 2021 CKD-EPI formula. The eGFR is a measure of the kidney's ability to filter waste from the blood, and is an important indicator of renal health.
		Endpoint
		
		The API endpoint for the eGFR calculation is:
		
		
		https://api.algomed.in/egfr1
		
		
		Input
		
		The API requires the following input parameters to be passed in the request body as a JSON object:
		* creatinine (float): The patient's creatinine level, in mg/dL. 
		* age (integer): The patient's age, in years.
		* Female (boolean): Is the patient a female, True or false.
		
		
		Input example:
		
		Here is an example for an input request 
		
		https://api.algomed.in/egfr1?creat=1.2&female=true&age=50
		
		Here the creatinine  is 1.2 mg/dl, The gender is female (Female = true) and age of the patient is 50 years. 
		
		curl -X 'GET' \
		  'https://api.algomed.in/egfr1?creat=1.2&female=true&age=50' \
		  -H 'accept: application/json'
		
		
		Output
		The API returns a JSON object with the following structure:
		
		{
		  "egfr": 55, ==>  The calculated eGFR value
		  "units": "mL/min/1.73m2",  ==> The units for the same
		  "CKD category": "G3a", ==> CKD category based on current guidelines 
		  "details": "This falls in category of mildly to moderately decreased eGFR" ==> Information about the CKD category
		}
		
		
		
		Example
		Here is an example of a request to the API endpoint:
		
		GET https://api.algomed.in/egfr1?creat=1.2&female=true&age=50
		
		And the corresponding response from the API:
		
		{
		  "egfr": 55,
		  "units": "mL/min/1.73m2",
		  "CKD category": "G3a",
		  "details": "This falls in category of mildly to moderately decreased eGFR"
		}
		
		
		Error Codes
		In case of any errors, the API returns a JSON object with the following structure:
		* error: A string indicating the error that occurred.
		* message: A detailed error message.
		Notes
		* The creatinine value must be entered in mg/dL.
		* The API only accepts requests with the 'application/json' content type.
		* The API returns a 500 Internal Server Error response in case of any internal errors.
		Disclaimer
		This API is provided for informational purposes only and is not intended to be used for medical advice, diagnosis, or treatment. The results of the eGFR calculation should always be interpreted in the context of a comprehensive medical evaluation by a qualified healthcare professional."
    
    """
    url = f"https://egfr-calculator-glomerular-filtration-rate-calculator.p.rapidapi.com/egfr1"
    querystring = {'age': age, 'creat': creat, 'female': female, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "egfr-calculator-glomerular-filtration-rate-calculator.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

