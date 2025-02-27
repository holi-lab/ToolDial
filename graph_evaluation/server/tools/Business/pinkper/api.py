import requests
import json
from datetime import date, datetime, timedelta
import os

from typing import Optional, Dict, Union, List


def pink_per(toolbench_rapidapi_key: str='088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff'):
    """
    "This is PinkPer, which is one of the leading brands specializing in paper cut light boxes, 3d shadow boxes and 3d silhouette light box template, glossy boxes and more recently in cutter and cricut designs. Over 300 free SVG files with designs from talented designers are added every week for everyone to choose from, If you are a handicraft lover, come to us
		Website: https://pinkper.com/ 
		Address: 116 Me Tri Ha Street, Nam Tu Liem District, Hanoi City
		Google map: https://www.google.com/maps?cid=967296933397643637 
		Phone: 0333740864
		Email:baoanhbox@gmail.com
		Social:
		+ https://www.facebook.com/Pinkper-108885481699507 
		+ https://www.instagram.com/pinkper143/ 
		+ https://www.pinterest.com/PinkperPappercut/_created/ 
		+ https://www.tiktok.com/@pinkperpappercut?lang=vi-VN 
		+ https://twitter.com/pinkpercom
		+ https://www.linkedin.com/in/pinkpercom/
		+https://www.youtube.com/channel/UCa5ZLSEzD_L6zuMQYZbo-wA/about
		https://pinkper.com/shop/ 
		https://pinkper.com/danh-muc-san-pham/family/ 
		https://pinkper.com/danh-muc-san-pham/love/ 
		https://www.google.com/maps?cid=967296933397643637
		https://experiment.com/users/pinkpercom
		https://letterboxd.com/pinkpercom/
		https://sketchfab.com/pinkpercom
		https://www.pozible.com/profile/pinkpercom
		https://www.reverbnation.com/artist/pinkpercom
		https://www.sqlservercentral.com/forums/user/pinkpercom
		https://yolotheme.com/forums/users/pinkpercom/
		http://bioimagingcore.be/q2a/user/pinkpercom
		http://maisoncarlos.com/UserProfile/tabid/42/UserID/1035253/Default.aspx
		http://myboxmoving.com/UserProfile/tabid/43/UserID/612493/Default.aspx
		http://vnvista.com/forums/member94294.html
		http://www.apelondts.org/Activity-Feed/My-Profile/UserId/17307
		http://www.articledude.com/classifieds/user/profile/624349
		http://www.conejousd.org/sequoia/User-Profile/UserId/121030
		http://www.synthedit.com/qa/user/pinkpercom
		http://www.worldchampmambo.com/UserProfile/tabid/42/UserID/231267/Default.aspx
		https://440hz.my/forums/users/pinkpercom/
		https://alexathemes.net/forums/users/ceoasd77/
		https://australian-school-holidays.mn.co/members/10671975
		https://band.us/band/87143227/intro
		https://bikepgh.org/message-board/users/pinkpercom/
		https://bittube.video/a/pink_per/video-channels
		https://code.sinergia-errors.ini.uzh.ch/pinkpercom
		https://community.growthhackers.com/members/10672065?agree=true
		https://developers.oxwall.com/user/pinkpercom
		https://fileforum.com/profile/pinkpercom
		https://flythemes.net/forums/users/pinkpercom/
		https://forum.honorboundgame.com/user-361642.html
		https://forum.umbandaeucurto.com/usuario/pinkpercom
		https://forums.prosportsdaily.com/member.php?1187305-pinkpercom
		https://fyi.org.nz/user/pinkpercom
		https://gaiauniversity.org/members/pinkpercom/profile/
		https://gettogether.community/profile/30449/
		https://gifyu.com/pinkpercom
		https://gotartwork.com/Profile/pink-per/136010/
		https://hashnode.com/@pinkpercom
		https://itseovn.com/members/pinkpercom.53247/
		https://k289gitlab1.citrin.ch/pinkpercom
		https://kwafoo.coe.neu.edu:7788/git/pinkpercom
		https://lab.quickbox.io/pinkpercom
		https://learn.acloud.guru/profile/pinkpercom
		https://my.archdaily.com/us/@pinkpercom
		https://my.olympus-consumer.com/members/pinkpercom
		https://myopportunity.com/profile/pink-per/sl
		https://osf.io/7kq85/
		https://paper.li/~/publisher/78f15d4a-8350-4278-bf86-9f0839530658
		https://pastelink.net/zu5ikyr9
		https://postheaven.net/pinkpercom/pink-per
		https://radiovybe.com/pinkpercom
		https://reedsy.com/discovery/user/pinkper
		https://rispondipa.it/user/pinkpercom
		https://scorestream.com/user/pinkpercom-2505795
		https://skitterphoto.com/photographers/31485/pinkpercom
		https://starity.hu/profil/317841-pinkpercom/
		https://stevewestenra.com/community/profile/pinkpercom/
		https://stocktwits.com/pinkpercom
		https://storium.com/user/pinkpercom
		https://pinkpercom.imgbb.com/
		https://pinkpercom.netboard.me/
		https://pinkpercom.ukit.me/
		https://support.advancedcustomfields.com/forums/users/pinkpercom/
		https://support.themecatcher.net/forums/users/pinkpercom
		https://www.anphabe.com/profile/pink.per
		http://pyttkvtphcm.gov.vn/question/pinkper/"
    
    """
    url = f"https://pinkper.p.rapidapi.com/"
    querystring = {}
    
    headers = {
            "X-RapidAPI-Key": toolbench_rapidapi_key,
            "X-RapidAPI-Host": "pinkper.p.rapidapi.com"
        }


    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:
        observation = response.text
    return observation

