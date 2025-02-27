import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def jsc_result(year: str, board: str, roll: str, exam: str, reg: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Endpoint URL: https://script.google.com/macros/s/AKfycbxxbo-f1KJgrOKPsl4W_WSACXLz188SRmhCBxdNq0oWDs4zTPgDvPKffMAkglq6ozzj/exec
		
		Endpoint Method: GET
		
		Parameters:
		
		exam (required): Specifies the examination type. Use "jsc" for Junior School Certificate (JSC) result.
		year (required): Specifies the year of the examination.
		board (required): Specifies the education board for which the result is desired.
		roll (required): Specifies the roll number of the student.
		reg (optional): Specifies the registration number of the student (if applicable).
		Example Usage:
		GET https://script.google.com/macros/s/AKfycbxxbo-f1KJgrOKPsl4W_WSACXLz188SRmhCBxdNq0oWDs4zTPgDvPKffMAkglq6ozzj/exec?exam=jsc&year=2008&board=comilla&roll=306737&reg=630621
		
		Description:
		The JSC Result endpoint allows you to retrieve the Junior School Certificate (JSC) examination results for individual students from the specified education board. By providing the necessary parameters, you can access detailed information about a student's performance in the JSC examination for a specific year.
		
		exam: Specify "jsc" to indicate that you want to retrieve the JSC result.
		year: Provide the year of the examination to retrieve the corresponding results.
		board: Specify the education board for which you want to fetch the result. For example, "comilla" represents the Comilla Education Board.
		roll: Enter the roll number of the student whose result you want to retrieve.
		reg (optional): If applicable, provide the registration number of the student to narrow down the search and retrieve the result more accurately.
		The endpoint will return the JSC result in a structured format, providing subject-wise marks, grade, and overall standings for the specified student. You can integrate this endpoint into your application or system to automate the retrieval and processing of JSC results for various purposes, such as student performance tracking, academic analysis, and reporting.
		
		Please note that you need to replace the example parameter values (year, board, roll, reg) in the URL with the actual values you want to query."
    
    """
    url = f"https://education-board-bangladesh-result.p.rapidapi.com/"
    querystring = {'year': year, 'board': board, 'roll': roll, 'exam': exam, 'reg': reg, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "education-board-bangladesh-result.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

