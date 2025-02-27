import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def colleges_api(state: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All U.S. colleges. Includes: College name, college rating, number of years, Grade, cost, number of students, ratings for academics, diversity, athletics, professors, dorms, student life, value, campus, party scene, location, food and safety. Also includes a photo of each school."
    
    """
    url = f"https://hotciti.p.rapidapi.com/getCollegesData/{state}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotciti.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def places_to_live_addon(experience_type: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Make your places to live data even better with location picture links, location video links, and local town website links (where available).  View and example here: https://hotciti.com/singlelisting?map-search1=Destin+Florida"
    
    """
    url = f"https://hotciti.p.rapidapi.com/getExperiencesData/{experience_type}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotciti.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def places_to_live(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Places to live API includes 42,000 U.S. cities, over 100 data points per location, covering topics like population, household information, cost of living,  weather, crime & safety, employment, taxes, homes status, educational attainment,  school system, commute time, income, religious affiliation, occupation, and politics. Also, ask about our upgrade which includes pictures of every location, videos of every location, and URLs to town websites.
		ID	
		City	City name, example  "Boston"
		City_Slug	City name small characters, example "boston"
		State_Abbreviated	State abbreviation, example "MA"
		State	Full state name, example "Massachusetts"
		City_State_Combined	City & State combined, example, "Boston Massachusetts"
		County	Group of towns, example "Worcester" county
		Region	Area of the country, example: "North East"
		Latitude	Locational latitude of a city
		Longitude	Location Longitude of a city
		Nearest_Major_City	The nearest major city (more than 100k residents) to a given town
		Zip_Code	Zipcode
		Population	Number of residents in a city
		Pop__Density	Number of residents per square mile
		Pop__Change	Annual percentage of population increase, decrease
		Median_Age	Average age of residents in a particular city
		Households	The number of homes in a particular city
		Household_Size	The average number of residents in a home, per city
		Male_Population	The number of male residents in a city
		Female_Population	The number of female residents in a city
		Married_Population	Percentage of the population that is married
		Single_Population	Percentage of the population that is not married
		Air_Quality	The quality of the air in a particular city
		Water_Quality_Score	The quality of the water in a particular city
		Physicians_per_100k	Number of doctors per 100k residents
		Unemployment_Rate	The average unemployment rate in a particular city
		Recent_Job_Growth	The percentage of added jobs in a particular city
		Future_Job_Growth	Projections for future jobs in a particular city
		Sales_Taxes	The sales tax rate in a particular city (Percentage)
		Income_Taxes	The income tax rate in a particular city (Percentage)
		Income_per_Cap_	amount of money earned per person in a particular city
		Household_Income	amount of money earned per household in a particular city
		Income_Less_Than_15K	Percentage of Homes with an income less than 15,000
		Income_between_15K_and_25K	Percentage of Homes with an income between 15,000 and 25,000
		Income_between_25K_and_35K	Percentage of Homes with an income between 25,000 and 35,000
		Income_between_35K_and_50K	Percentage of Homes with an income between 35,000 and 50,000
		Income_between_50K_and_75K	Percentage of Homes with an income between 50,000 and 75,000
		Income_between_75K_and_100K	Percentage of Homes with an income between 75,000 and 100,000
		Income_between_100K_and_150K	Percentage of Homes with an income between 100,000 and 150,000
		Income_between_150K_and_250K	Percentage of Homes with an income between 150,000 and 250,000
		Income_between_250K_and_500K	Percentage of Homes with an income between 250,000 and 500,000
		Income_greater_than_500K	Percentage of Homes with an income greater than 500,000
		Management_Business_and_Financia	Percentage of residents who work in business and financial
		Professional_and_Related_Occupat	Percentage of residents who work professional or related fields
		Sales_and_Office	Percentage of residents who work in sales or inner office
		Service	Percentage of residents who work in service positions
		Farming_Fishing_and_Forestry	Percentage of residents who work in fishing, farming or forestry
		Construction_Extraction_and_Main	Percentage of residents working construction or related fields
		Production_Transportation_and_Ma	Percentage of residents working in production, transportation and related fields
		Median_Home_Age	Median age of homes in a particular city
		Median_Home_Cost	Median home cost in a particular city
		Home_Appreciation	The increase of a home's value over time in a particular city  (Percentage)
		Homes_Owned	the percentage of private homes in a particular city
		Housing_Vacant	The percentage of vacant homes in a particular city
		Homes_Rented	The percentage of rented homes in a particular city
		Property_Tax_Rate	The property tax rate in a particular city (Percentage)
		Less_Than_20000	Percentage of homes valued less than 200,000
		a20000_to_39999	Percentage of homes valued between
		a40000_to_59999	Percentage of homes valued between
		a60000_to_79999	Percentage of homes valued between
		a80000_to_99999	Percentage of homes valued between
		a100000_to_149999	Percentage of homes valued between
		a150000_to_199999	Percentage of homes valued between
		a200000_to_299999	Percentage of homes valued between
		a300000_to_399999	Percentage of homes valued between
		a400000_to_499999	Percentage of homes valued between
		a500000_to_749999	Percentage of homes valued between
		a1000000_or_more	Percentage of homes valued between
		a1999_to_October_2005	Percentage of homes built between 1999 and 2005
		a1995_to_1998	Percentage of homes built between 1995 and 1998
		a1990_to_1994	Percentage of homes built between 1990-1994
		a1980_to_1989	Percentage of homes built between 1980-1989
		a1970_to_1979	Percentage of homes built between 1970-1979
		a1960_to_1969	Percentage of homes built between 1960-1969
		a1950_to_1959	Percentage of homes built between 1950-1959
		a1940_to_1949	Percentage of homes built between 1940-1949
		a1939_or_Earlier	Percentage of homes built in 1939 or earlier
		Violent_Crime	Violent crime score (lower is better)
		Property_Crime	Property crime score (lower is better)
		Rainfall_in_	Annual rainfall in inches
		Snowfall_in_	Annual snowfall in inches
		Precipitation_Days	Combined days of rain or snow annually
		Sunny_Days	Number of sunny days annually
		Avg__July_High	Avg. high temperature in July
		Avg__Jan__Low	Avg. low teperature in January
		Comfort_Index_higherbetter	Environmental suitability score (17 - 97 higher is better)
		UV_Index	Ultraviolet index measures sunburn producing radiation (range 5-68)
		Elevation_ft_	Overall elevation in a particular city (measured in feet)
		School_Annual_Spend_Per Student	The average amount spent on a single student in a particular city
		Student__Teacher_Ratio	the number of students per teacher in a particular city
		Students_per_Librarian	The number of students per librarians in a particular city
		Students_per_Counselor	The number of students per counselors in a particular city
		a2_yr_College_Grad_	the percentage of residents with a 2 yr college degree
		a4_yr_College_Grad_	the percentage of residents with a 4 yr college degree
		Graduate_Degrees	the percentage of residents with a graduate degree
		High_School_Grads_	the percentage of residents with H.S. diploma
		Mass_Transit	the percentage of residents who commute via mass transit
		Auto_alone	the percentage of residents who commute via automobile
		Commute_Time	the average commute time to work for a particular city
		Carpool	the percentage of residents who carpool to work in a particular city
		Work_at_Home	The percentage of residents who work from home in a particular city
		Commute_Less_Than_15_min_	The percentage of residents who have a commute time of less than 15 minutes
		Commute_15_to_29_min_	The percentage of residents who have a commute time of 15 - 29 minutes
		Commute_30_to_44_min_	The percentage of residents who have a commute time of 30-44 minutes
		Commute_45_to_59_min_	The percentage of residents who have a commute time of 45 to 59 minutes
		Commute_greater_than_60_min_	the percentage of residents with a commute time of greater than 60 minutes
		Overall_Cost_Of_Living	The overall cost of living score (avg. score is 100, over 100 is more expensive)
		Food	Score the cost of buying food in a particular city (avg. score is 100)
		Utilities	Score the cost of utilities in a particular city (avg. score is 100)
		Miscellaneous	Avg. Cost of miscellaneous items in a particular city (avg. score is 100)
		Percent_Religious	Percentage of residents who claim religious affiliation
		Catholic	Percentage of residents that identify as Catholic
		Protestant	Percentage of residents that identify as Protestant
		LDS	Percentage of residents that identify as LDS
		Episcopalian	Percentage of residents that identify as Episcopal
		Baptist	Percentage of residents that identify as Baptist
		Pentecostal	Percentage of residents that identify as Pentecostal
		Lutheran	Percentage of residents that identify as Lutheran
		Methodist	Percentage of residents that identify as Methodist
		Presbyterian	Percentage of residents that identify as Presbyterian
		Other_	Percentage of residents that identify as other
		Christian	Percentage of residents that identify as Christian
		Jewish	Percentage of residents that identify as Jewish
		Islam	Percentage of residents that identify as Islam
		Democrat	Percentage of residents registered Independent or Other
		Republican	Percentage of residents registered Democrat
		Independent_Other Percentage of residents registered Independent"
    
    """
    url = f"https://hotciti.p.rapidapi.com/getLocationsData/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotciti.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def hotels_api(location: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "All U.S. hotels within a 20-mile radius of the entered location. Includes address, pictures, rating, costs, check in time, number of rooms, checkout time, year opened, number of floors, and a description of the hotel."
    
    """
    url = f"https://hotciti.p.rapidapi.com/getHotelsData/{location}"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "hotciti.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

