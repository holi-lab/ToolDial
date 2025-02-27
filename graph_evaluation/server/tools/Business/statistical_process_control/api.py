import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def westgard_details(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "From [Wikipedia](https://en.wikipedia.org/wiki/Westgard_rules):
		
		> The Westgard rules are a set of statistical patterns, each being unlikely to occur by random variability, thereby raising a suspicion of faulty accuracy or precision of the measurement system. They are used for laboratory quality control, in "runs" consisting of measurements of multiple samples. They are a set of modified Western Electric rules, developed by James Westgard and provided in his books and seminars on quality control."
    
    """
    url = f"https://statistical-process-control1.p.rapidapi.com/spc/rules/westgard"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "statistical-process-control1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def western_electric_details(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "From [Wikipedia](https://en.wikipedia.org/wiki/Western_Electric_rules):
		
		> The Western Electric rules are decision rules in statistical process control for detecting out-of-control or non-random conditions on control charts.[1] Locations of the observations relative to the control chart control limits (typically at ±3 standard deviations) and centerline indicate whether the process in question should be investigated for assignable causes. The Western Electric rules were codified by a specially-appointed committee of the manufacturing division of the Western Electric Company and appeared in the first edition of a 1956 handbook, that became a standard text of the field.[2] Their purpose was to ensure that line workers and engineers interpret control charts in a uniform way."
    
    """
    url = f"https://statistical-process-control1.p.rapidapi.com/spc/rules/western-electric"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "statistical-process-control1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def continuous_variable_details(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Continuous value control charts are graphical displays of quality characteristics that have been measured or computed from a sample versus the sample number or time. Furthermore, these control charts contain a center line representing the average value of the quality characteristics and two other horizontal lines known as upper control limit (UCL) and lower control limit (LCL) ."
    
    """
    url = f"https://statistical-process-control1.p.rapidapi.com/spc/control/variable"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "statistical-process-control1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def discrete_attribute_details(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Per [iSixSigma](https://www.isixsigma.com/tools-templates/control-charts/steps-in-constructing-a-c-chart/):
		
		> Attribute control charts arise when items are compared with some standard and then are classified as to whether they meet the standard or not. The control chart is used to determine if the rate of nonconforming product is stable and detect when a deviation from stability has occurred. The argument can be made that a LCL should not exist, since rates of nonconforming product outside the LCL is in fact a good thing; we WANT low rates of nonconforming product. However, if we treat these LCL violations as simply another search for an assignable cause, we may learn for the drop in nonconformities rate and be able to permanently improve the process."
    
    """
    url = f"https://statistical-process-control1.p.rapidapi.com/spc/control/attribute"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "statistical-process-control1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def nelson_details(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "From [Wikipedia](https://en.wikipedia.org/wiki/Nelson_rules):
		
		> Nelson rules are a method in process control of determining whether some measured variable is out of control (unpredictable versus consistent). Rules for detecting "out-of-control" or non-random conditions were first postulated by Walter A. Shewhart in the 1920s. The Nelson rules were first published in the October 1984 issue of the Journal of Quality Technology in an article by Lloyd S Nelson."
    
    """
    url = f"https://statistical-process-control1.p.rapidapi.com/spc/rules/nelson"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "statistical-process-control1.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

