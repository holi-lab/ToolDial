import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def match(birthdetails: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Accepts a string parameter containing the first person's name and birth date, followed by up to 10 other people and their birth dates  with which to apply the astrological matching algorithms and return them in order of most to least compatible."
    
    """
    url = f"https://starlovematch.p.rapidapi.com/api/"
    querystring = {'birthdetails': birthdetails, }
    if birthdetails:
        querystring['birthdetails'] = birthdetails
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starlovematch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def brad_pitt(birthdetails: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a demo endpoint showing how it processes a list of Jennifer Aniston's partners.
		
		Here are the input 'birthdetails' parameters :  name=Brad Pitt&dob=12/18/1963&name1=Angelina Jolie&dob1=06/04/1975&name2=Jennifer Aniston&dob2=02/11/1969&name3=Gwyneth Paltrow&dob3=9/27/1972&name4=Charlize Theron&dob4=8/07/1975&sort=O&NC=C&ryr=2022&details=N&coupon=12345678
		
		You will see that it calculates Nicole Poturalski as his best overall (sort='O')  general match.
		
		If you change the input parameters and specify sort='P' (Physical), you will see that she is also currrently (20220 his best physical (sex) match.
		
		If you wish to see how it sorts his Intellectual matches, change this to sort='I', and you will see that Jennifer Aniston is currently her best intellectual partner.
		
		For interest, change the ryr parameter to '2004' using the sortorder = 'L'ove setting, you'll see how Jennifer was at the top of the love list, i.e.
		
		name=Brad Pitt&dob=12/18/1963&name1=Angelina Jolie&dob1=06/04/1975&name2=Jennifer Aniston&dob2=02/11/1969&name3=Gwyneth Paltrow&dob3=9/27/1972&name4=Charlize Theron&dob4=8/07/1975&name5=Nicole Poturalski&dob5=01/02/1993&sort=L&NC=C&ryr=2000&details=N&coupon=12345678
		
		
		To have a look at what his compatibility ratings were in 2015, enter the following parameters : 
		
		name=Brad Pitt&dob=12/18/1963&name1=Angelina Jolie&dob1=06/04/1975&name2=Jennifer Aniston&dob2=02/11/1969&name3=Gwyneth Paltrow&dob3=9/27/1972&name4=Charlize Theron&dob4=8/07/1975&name5=Nicole Poturalski&dob5=01/02/1993&sort=O&NC=C&ryr=2015&details=N&coupon=12345678
		
		You'll see Angeline Jolie is second from the bottom, with a negative overall rating.
		
		Negative ratings indicate that there are bad aspects in operation and this person should be avoided. In 2015 Brad and Angelina divorced.
		
		But there is some good news for him and his family. If you look ahead to the year 2030 by changing the ryr parameter to 2030, you will see he has a strong love relationship with Angelina Jolie again, and Jennifer Aniston has faded away in his life.
		
		name=Brad Pitt&dob=12/18/1963&name1=Angelina Jolie&dob1=06/04/1975&name2=Jennifer Aniston&dob2=02/11/1969&name3=Gwyneth Paltrow&dob3=9/27/1972&name4=Charlize Theron&dob4=8/07/1975&name5=Nicole Poturalski&dob5=01/02/1993&sort=O&NC=C&ryr=2030&details=N&coupon=12345678
		
		Of course, this is looking too far ahead to be generally useful at this point in time, so it is better to use the current year for the ryr parameter."
    
    """
    url = f"https://starlovematch.p.rapidapi.com/api/"
    querystring = {'birthdetails': birthdetails, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starlovematch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

def jennifer_aniston(birthdetails: str, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is a demo endpoint showing how it processes a list of Jennifer Aniston's partners.
		
		Here are the input 'birthdetails' parameters :  name=Jennifer Aniston&dob=2/11/1969&name1=Justin Theroux&dob1=8/10/1971&name2=Gerard Butler&dob2=11/13/1969&name3=John Mayer&dob3=10/16/1977&name4=Paul Sculfor&dob4=2/1/1971&name5=David Schwimmer&dob5=11/2/1969&name6=Vince Vaughn&dob6=3/28/1970&name7=Brad Pitt&dob7=12/18/1963&name8=Paul Rudd&dob8=4/6/1969&name9=Tate Donovan&dob9=9/25/1963&name10=Daniel McDonald&dob10=7/30/1960&sort=S&NC=C&ryr=2022&details=N&coupon=12345678
		
		You will see that it calculates Paul Rudd to be her best overall (sort='O')  general match, followed by Justin Theroux and Brad Pitt.
		
		However, if you change the input parameters and specify sort='P' (Physical), you will see that John Mayer is her best physical (sex) match.
		
		Likewise, if you wish to see how it calculates her Intellection matches in order, change this to sort='I', and you will see that Paul Rudd is again her best intellectual partner."
    
    """
    url = f"https://starlovematch.p.rapidapi.com/api/"
    querystring = {'birthdetails': birthdetails, }
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "starlovematch.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

