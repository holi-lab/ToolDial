import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def showgirl_live(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "Showgirl live là ứng dụng dành riêng cho điện thoại, có chức năng xem idol live kiếm tiền thoải mái, tải app show girl miễn phí.
		#showgirl apk #showgirl ios #app live #Showgirl
		Địa chỉ 5 Giải Phóng, Đồng Tâm, Hai Bà Trưng, Hà Nội
		Email:  showgirlapp1@gmail.com
		Hotline  0865177667
		Website  https://showgirlapp.live/ 
		Google map 
		https://g.page/r/CUHW22-wLwnwEBA  
		https://showgirlapp.live/tai-ve/showgirl-apk/ 
		https://showgirlapp.live/tai-ve/showgirl-apk/ 
		https://showgirlapp.live/thu-muc/app-live/ 
		https://sites.google.com/view/showgirlapp 
		https://www.helpforenglish.cz/profile/235037-showgirl
		https://android.libhunt.com/u/showgirl
		https://twinoid.com/user/10035006
		http://recipes.mentaframework.org/user/edit/180571.page
		https://d.cosx.org/u/showgirl
		https://vhearts.net/showgirl
		https://roomstyler.com/users/showgirlapp
		http://qooh.me/showgirlapp
		http://onlineboxing.net/jforum/user/editDone/150849.page
		https://dev.funkwhale.audio/showgirl
		http://phillipsservices.net/UserProfile/tabid/43/userId/125010/Default.aspx
		https://bittube.tv/profile/showgirl
		https://community.stadia.com/t5/user/viewprofilepage/user-id/92580
		https://git.feneas.org/showgirl
		https://www.godryshop.it/members/showgirl
		https://yemle.com/profile/showgirl
		https://forum.singaporeexpats.com/memberlist.php?mode=viewprofile&u=364123
		https://www.feedsfloor.com/profile/showgirl
		https://www.woddal.com/showgirl
		https://sallatunturinkoulu.purot.net/profile/showgirl
		https://storium.com/user/showgirl
		https://worldbeyblade.org/User-showgirl
		https://www.drupalgovcon.org/user/110911
		https://liveviewsports.com/community/profile/showgirl/
		https://gettogether.community/profile/28671/
		https://www.phuot.vn/members/showgirl.268342/#about
		http://notes.soliveirajr.com/user/editDone/44310.page
		https://www.ohay.tv/profile/showgirl
		https://www.bigpictureclasses.com/users/showgirl"
    
    """
    url = f"https://showgirl-live.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "showgirl-live.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

