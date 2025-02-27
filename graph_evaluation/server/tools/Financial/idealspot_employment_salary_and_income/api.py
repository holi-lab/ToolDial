import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def get_the_salary_wage_per_employee_per_annum(location: str, version: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get salary/wage data per employee per annum within a specified geo location."
    location: Represents a buffer, region, or custom polygon specification. Accepts the Location model (as a Buffer, Region, or Custom Polygon) formatted as a JSON string. Multiple location query parameters are allowed. NOTE: When requesting multiple locations, you must include brackets(i.e. ?location[]=...&location[]=...). If not included, only the last location will be used.
        
    """
    url = f"https://idealspot-employment-salary-and-income.p.rapidapi.com/api/v1/data/insights/salaries/query"
    querystring = {'location': location, 'version': version, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "idealspot-employment-salary-and-income.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_the_house_hold_income(version: str, location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get the total amount all members of a household earn. within a specified geo location. Includes Average Household Income, Median Household Income, Number Households, and Binned Income Ranges from $0 to $500,000+"
    location: Represents a buffer, region, or custom polygon specification. Accepts the Location model (as a Buffer, Region, or Custom Polygon) formatted as a JSON string. Multiple location query parameters are allowed. NOTE: When requesting multiple locations, you must include brackets(i.e. ?location[]=...&location[]=...). If not included, only the last location will be used.
        
    """
    url = f"https://idealspot-employment-salary-and-income.p.rapidapi.com/api/v1/data/insights/household-income/query"
    querystring = {'version': version, 'location': location, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "idealspot-employment-salary-and-income.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def get_resident_s_occupation(location: str, version: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Get occupations of residents within a specified geo location. The following occupations categories are supported:
		
		- Advertising, marketing, promotions, public relations, and sales managers
		- Agricultural workers
		- Air transportation occupations
		- All other building and grounds cleaning and maintenance workers
		- All other counselors, social, and religious workers
		- All other legal and related workers
		- And mobile equipment mechanics, installers, and repairers
		- Animal care and service workers
		- Architects, surveyors, and cartographers
		- Architecture and engineering occupations
		- Art and design occupations
		- Arts, design, entertainment, sports, and media occupations
		- Assemblers and fabricators
		- Blue Collar
		- Building and grounds cleaning and maintenance occupations
		- Building cleaning and pest control workers
		- Business and financial operations occupations
		- Business operations specialists
		- Communications equipment operators
		- Community and social services occupations
		- Computer and mathematical science occupations
		- Computer specialists
		- Construction and extraction occupations
		- Construction trades and related workers
		- Cooks and food preparation workers
		- Counselors, social workers, and other community and social service specialists
		- Drafters, engineering, and mapping technicians
		- Education, training, and library occupations
		- Electrical and electronic equipment mechanics, installers, and repairers
		- Engineers
		- Entertainers and performers, sports and related occupations
		- Entertainment attendants and related workers
		- Extraction workers
		- Farming, fishing, and forestry occupations
		- Financial clerks
		- Financial specialists
		- Fire fighting and prevention workers
		- First-line supervisors/managers, protective service workers
		- Fishing and hunting workers
		- Food and beverage serving workers
		- Food preparation and serving related occupations
		- Food processing occupations
		- Forest, conservation, and logging workers
		- Funeral service workers
		- Grounds maintenance workers
		- Health diagnosing and treating practitioners
		- Health technologists and technicians
		- Healthcare practitioners and technical occupations
		- Healthcare support occupations
		- Helpers, construction trades
		- Information and record clerks
		- Installation, maintenance, and repair occupations
		- Law enforcement workers
		- Lawyers, judges, and related workers
		- Legal occupations
		- Legal support workers
		- Librarians, curators, and archivists
		- Life scientists
		- Life, physical, and social science occupations
		- Life, physical, and social science technicians
		- Management occupations
		- Material moving occupations
		- Material recording, scheduling, dispatching, and distributing occupations
		- Mathematical science occupations
		- Media and communication equipment occupations
		- Media and communication occupations
		- Metal workers and plastic workers
		- Military
		- Motor vehicle operators
		- Nursing, psychiatric, and home health aides
		- Occupational and physical therapist assistants and aides
		- Office and administrative support occupations
		- Operations specialties managers
		- Other construction and related workers
		- Other education, training, and library occupations
		- Other food preparation and serving related workers
		- Other healthcare practitioners and technical occupations
		- Other healthcare support occupations
		- Other installation, maintenance, and repair occupations
		- Other management occupations
		- Other office and administrative support workers
		- Other personal care and service workers
		- Other production occupations
		- Other protective service workers
		- Other sales and related workers
		- Other teachers and instructors
		- Other transportation workers
		- Personal appearance workers
		- Personal care and service occupations
		- Physical scientists
		- Plant and system operators
		- Postsecondary teachers
		- Primary, secondary, and special education teachers
		- Printing occupations
		- Production occupations
		- Protective service occupations
		- Rail transportation occupations
		- Religious workers
		- Retail sales workers
		- Sales and related occupations
		- Sales representatives, services
		- Sales representatives, wholesale and manufacturing
		- Secretaries and administrative assistants
		- Social scientists and related occupations
		- Supervisors of installation, maintenance, and repair workers
		- Supervisors, building and grounds cleaning and maintenance workers
		- Supervisors, construction and extraction workers
		- Supervisors, farming, fishing, and forestry workers
		- Supervisors, food preparation and serving workers
		- Supervisors, office and administrative support workers
		- Supervisors, personal care and service workers
		- Supervisors, production workers
		- Supervisors, sales workers
		- Supervisors, transportation and material moving workers
		- Textile, apparel, and furnishings occupations
		- Top executives
		- Transportation and material moving occupations
		- Transportation, tourism, and lodging attendants
		- Unclassified
		- Water transportation occupations
		- White Collar
		- Woodworkers"
    location: Represents a buffer, region, or custom polygon specification. Accepts the Location model (as a Buffer, Region, or Custom Polygon) formatted as a JSON string. Multiple location query parameters are allowed. NOTE: When requesting multiple locations, you must include brackets(i.e. ?location[]=...&location[]=...). If not included, only the last location will be used.
        
    """
    url = f"https://idealspot-employment-salary-and-income.p.rapidapi.com/api/v1/data/insights/occupation/query"
    querystring = {'location': location, 'version': version, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "idealspot-employment-salary-and-income.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

