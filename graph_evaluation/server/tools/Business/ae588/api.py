import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def ae888(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "AE888 - AE3888 là nhà cái số 1 hiện nay, cung cấp game đa dạng như đá gà, thể thao, lô đề online, tài xỉu. Ae588 ủy quyền Ae888 cung cấp link vào nhà cái.
		#Ae888 #ae88 #tructiepdaga #ae588net  #dangkyae888 /m/033_l8  /m/033_l8 #Casino 
		Hotline 0924.665.790
		Địa chỉ : 345 Đường Xuân Phương Xuân Phương,Từ Liêm,Hà Nội
		EMail: ae588net@gmail.com
		Website https://ae588.net/ 
		https://ae588.net/ae88-nha-cai-ca-cuoc-truc-tuyen-ae588/ 
		https://ae588.net/da-ga-truc-tiep/ 
		https://ae588.net/ae888-dang-ky-huong-dan-dang-ky-tai-khoan-ca-cuoc-ae888/ 
		https://en.wikipedia.org/wiki/Online_casino 
		https://www.facebook.com/ae588net 
		https://sites.google.com/view/ae588net/ae888 
		https://ae588net.blogspot.com/ 
		https://www.allmyfaves.com/ae588net
		https://www.metroflog.co/ae588
		http://www.effecthub.com/people/ae588net
		descubre.beqbe.com/p/ae588
		https://www.mapleprimes.com/users/ae588net
		https://www.bahamaslocal.com/userprofile/117019/ae588net.html
		http://www.good-tutorials.com/users/ae588
		https://yarabook.com/ae588
		https://www.bitsdujour.com/profiles/XYLqiX
		https://www.giantbomb.com/profile/ae588/
		https://www.techrum.vn/members/ae588.188286/
		https://www.facer.io/u/ae588net
		https://ae588.netboard.me/
		https://socialcompare.com/en/member/ae588-6cazjc8j
		https://git.qoto.org/ae588
		https://www.bbuzzart.com/profile/427401
		http://wiznotes.com/UserProfile/tabid/84/userId/1522211/Default.aspx
		https://englishbaby.com/findfriends/gallery/detail/2390818
		http://kiredu.ru/UserProfile/tabid/182/userId/77394/Default.aspx
		https://www.thebranfordgroup.com/dnn3/UserProfile/tabid/214/UserId/59459/Default.aspx
		https://beermapping.com/account/ae588
		http://forums.qrecall.com/user/editDone/286742.page
		http://projectcs.sci.ubu.ac.th/ae588
		http://battlebrothersgame.com/forums/users/ae588net/
		https://www.bitrated.com/ae588"
    
    """
    url = f"https://ae588.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "ae588.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

