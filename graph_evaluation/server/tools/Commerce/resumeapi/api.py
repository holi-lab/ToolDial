import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def imtired(two: str=None, six: str=None, ten: str=None, seven: str=None, three: str=None, eleven: str=None, nine: str=None, five: str=None, four: str=None, eight: str=None, one: str=None, toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "imtiredss sada sd asdas"
    two: Experienced and results-oriented Store Manager with three years of managerial experience in retail environments. Motivated professional with strengths in supervising employees and promoting optimal customer service. Adept in employee relations, inventory organization, and visual merchandising. Prepared, organized, and trained in product knowledge and store regulations. Committed to creating an environment that is conducive to achieving increased sales and customer satisfaction. 
        six: Experienced and dedicated Art Gallery Manager with several years of experience effectively managing galleries and overseeing their day to day operations. Bringing forth a lifelong passion for art and creative expression. Adept in networking with artists, planning exhibitions, and organizing social and promotional events. Delivering unwavering leadership, organization, and valuable analytical thinking skills necessary to run a successful gallery. 
        ten: Experienced and professional IT Manager with over eight years of valuable experience in monitoring project performance to ensure company success. Familiar and adept in working with engineering, industrial engineering, change management, and business transformation systems. Reputation for continually achieving on time and under-budget performance goals. Accustomed to handling IT support, managing IT projects, and supporting various engineering departments with IT tools and applications. Bringing forth a motivated attitude and the ability to establish strong and productive relationships with other company members. 
        seven: Experienced Hair Stylist with over six years of experience cutting and styling hair. Adept in working with clients to achieve desired look and hair goals. Results oriented and able to work in a high traffic and fast paced environment. Dedicated to remaining up to date with the latest hair product and treatment offerings and technologies. Bringing forth a commitment to providing clients with the most pleasurable styling experience possible. 
        three: Experienced and passionate Caterer with over five years of experience in the catering industry, serving clients throughout New York, New Jersey, and Connecticut. I have catered for private clients, corporations, nonprofits, and cultural institutions. Striving to provide every client with innovation, quality, and commitment to excellent service. Specializing in American Cuisine and local flavors. Working to uphold a commitment to sustainable agriculture and environmentally friendly practices. 
        eleven: Experienced and self-motivated Office Secretary with ten+ years of industry experience overseeing the main offices of schools. Highly competent communicator skilled in multitasking and effectively communicating with others. Bringing forth a proven track record of successfully managing offices, and helping to lead school professionals to work toward reaching goals. 
        nine: Resourceful and dedicated High School Student with strong analytical skills and a serious work ethic. Bringing forth excellent organizational abilities, multitasking skills, and the drive to conquer goals. Adept at working well with others and committing myself as a positive team player. 
        five: Experienced Nail Technician with several years of experience providing quality nails services to clients. Dedicated to remaining up to date with the latest nail service technologies, designs, and trends. Bringing forth advanced nail technical abilities and the desire to provide clients with the best and most pleasurable salon experience possible.  
        four: Reliable and experienced Housekeeper with over five years of experience maintaining the cleanliness and organization of residences and offices. Skilled at performing basic cleaning duties, deep sanitation, organizing spaces, and tending to general housekeeping tasks in a time efficient manner. Committed to providing clients with superior service and satisfaction. 
        eight: Experienced and effective Operations Manager bringing forth valuable industry experience and a passion for management. Results oriented with a proven track record of improving overall operations within a company or department. Adept in analytical thinking, strategic planning, leadership, and the management of staff and procedures. 
        one: An ambitious web developer focused on front-end Ul/UX, with a penchant for creating innovative solutions to unsolvable problems. Passionate about modern design, unambiguous code, and thorough documentation. Goal-oriented and driven both as an individual and a team player. Seeking a position at a company that would benefit from my expertise in web development, enthusiasm, and creative flair. 
        
    """
    url = f"https://resumeapi.p.rapidapi.com/imtired"
    querystring = {}
    if two:
        querystring['two'] = two
    if six:
        querystring['six'] = six
    if ten:
        querystring['ten'] = ten
    if seven:
        querystring['seven'] = seven
    if three:
        querystring['three'] = three
    if eleven:
        querystring['eleven'] = eleven
    if nine:
        querystring['nine'] = nine
    if five:
        querystring['five'] = five
    if four:
        querystring['four'] = four
    if eight:
        querystring['eight'] = eight
    if one:
        querystring['one'] = one
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "resumeapi.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

