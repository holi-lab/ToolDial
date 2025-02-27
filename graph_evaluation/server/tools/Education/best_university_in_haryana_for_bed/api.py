import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def best_university_for_bed_in_haryana(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Which university's B.Ed. program is the best?
		It's hard to say which university's B.Ed. program is best. The university, on the other hand, is "The Best University" for a lot of other reasons. The factors may also differ from person to person, such as whether the university's location is more important to one person versus the university's fee or infrastructure to another. The year of establishment of the university, the courses offered, job assistance, and a host of other factors could also play a role. As a result, it's hard to provide a precise response to this.
		
		At the moment, eight universities in Haryana offer the B.Ed. program. These are:
		
		- Maharishi Dayanand University, Rohtak – MDU Rohtak 
		- Kurukshetra University, Kurukshetra – KUK Kurukshetra 
		- Deenbandhu Chhotu Ram University of Science and Technology, Murthal – DCRUST Murthal YMCA University of Science and Technology, Faridabad – YMCAUST Faridabad
		- Chaudhary Ranbir Singh University, Jind – CRSU Jind 
		- Geeta University
		
		
		MDU Rohtak is likewise certified as "A" grade college by NAAC. In 2016, MDU is ranked 44 by the Ministry of Human Resource Development of the National Government. Here, you can learn more about why MDU is the best school for a B.Ed. and about the MDU B.Ed.
		
		The Kurukshetra campus of the Haryana state government university known as KU Kurukshetra If you're coming from Delhi, KUK is about 160 kilometers away, on the way to Chandigarh. The campus of KUK covers 400 acres. NAAC also awarded KUK an "A" grade in 2009. The more data about KUK and why KUK is the best college for B.Ed and to be aware of the B.Ed course and affirmation prerequisite can be seen as here.
		
		The Deenbandhu Chhotu Ram University of Science and Technology – DCRUST Murthal, which is located in Murthal, Haryana, is one of the best universities known for offering technical courses to date. However, according to the government, "the Governor of Haryana hereby specifies that with effect from the academic session 2017-18, Deenbandhu Chhotu Ram University of Science and Technology, Murthal shall exercise its powers, over all B.Ed and Engineering Colleges (both Government and Non-Government) situated in the districts of Panipat and Sonipat." This change took effect in 2017 and continues through the present. You can find additional information about the government's decision here. We also hope that DCRUST Murthal will soon be one of the best universities in Haryana to offer a B.Ed. program due to their expertise academically and their rapid performance in the technical field. You can find additional information about DCRUST and the admissions process for the B.Ed.
		
		Geeta University have laid the runway for thousands of dreams taking off to the destination of careers with the journey. Geeta University is the top University in Haryana, with high placement records."
    
    """
    url = f"https://best-university-in-haryana-for-bed.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "best-university-in-haryana-for-bed.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

