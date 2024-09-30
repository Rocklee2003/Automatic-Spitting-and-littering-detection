import pywhatkit as pwk
import datetime

def sendImage(mobilenumber,imagepath,message):
    datet=str(datetime.datetime.now())
    st=datet.split(" ")
    kt=st[1].split(":")
    hourstr=kt[0]
    minstr=kt[1]
    hr=int(hourstr)
    min=int(minstr)
    if(min<40):
        min=min+1
    else:
        min=1
        hr=hr+1
    print(hr)
    print(min)
    
   
    pwk.sendwhats_image(mobilenumber, imagepath,message)
    
 
def sendMessageOnly(mobilenumber,message):
    datet=str(datetime.datetime.now())
    st=datet.split(" ")
    kt=st[1].split(":")
    hourstr=kt[0]
    minstr=kt[1]
    hr=int(hourstr)
    min=int(minstr)
    if(min<40):
        min=min+1
    else:
        min=1
        hr=hr+1
    print(hr)
    print(min)
    
    pwk.sendwhatmsg(mobilenumber,message,hr, min)
if __name__ == '__main__':
    sendImage()
    sendMessageOnly()
    