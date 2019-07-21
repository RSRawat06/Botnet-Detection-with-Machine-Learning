
import socket, struct, sys
import numpy as np
import pickle

def loaddata(fileName):
    """function to load the training and the test dataset"""

    file = open(fileName, 'r')

    xdata = []
    ydata = []
    xdataT = []
    ydataT = []
    count=0

    #dicts to convert protocols and state to integers
    protoDict = {'': 0}


    file.readline()

    liste= ["192.168.2.112",
            "198.164.30.2",
            "192.168.2.113",
            "192.168.2.112",
            "147.32.84.180",
            "147.32.84.140",
            "10.0.2.15",
            "172.16.253.130",
            "172.16.253.240",
            "192.168.3.35",
            "172.29.0.116",
            "192.168.248.165",
            "131.202.243.84",
            "192.168.2.110",
            "192.168.1.103",
            "192.168.2.109",
            "147.32.84.170",
            "147.32.84.130",
            "192.168.106.141",
            "172.16.253.131",
            "74.78.117.238",
            "192.168.3.25",
            "172.29.0.109",
            "10.37.130.4",
            "192.168.5.122",
            "192.168.4.118",
            "192.168.4.120",
            "192.168.2.105",
            "147.32.84.150",
            "147.32.84.160",
            "192.168.106.131",
            "172.16.253.129",
            "158.65.110.24",
            "192.168.3.65",
            "172.16.253.132" 
            ]


    for line in file:
        sd = line[:-1].split(',')
        check = 0
        sayac = 0
        while sayac < len(sd):
            sd[sayac] = sd[sayac].replace("\"", "")
            sayac = sayac + 1
        dur, proto, Sip, Dip, totB, label = sd[1], sd[4], sd[2], sd[3], sd[-2], sd[-1]
        temp = protoDict.get(proto, None)
        if temp==None:
            protoDict[proto] = count
            count=count+1
            print(protoDict[proto], proto)


    file = open(fileName, 'r')
    file.readline()
    count=0
    for line in file:
       sd = line[:-1].split(',')
       check=0
       sayac=0
       while sayac < len(sd):
          sd[sayac]=sd[sayac].replace("\"","")
          sayac=sayac+1
       dur, proto, Sip, Dip, totB, label = sd[1], sd[4], sd[2], sd[3], sd[-2], sd[-1]

       sayac=0
       while sayac < len(liste):
            if Sip == liste[sayac]:
                check=1
                break
            elif Dip == liste[sayac]:
                check = 1
                break
             
      
            sayac=sayac+1
        #back, nor, bot

       try:
           
           Sip = socket.inet_aton(Sip)
           Sip = struct.unpack("!L", Sip)[0]
       
       except:
           continue
       try:
           
           Dip = socket.inet_aton(Dip)
           Dip = struct.unpack("!L", Dip)[0]
       
       except:
           continue

       try:
           tempProto = protoDict[proto]
       except:
           tempProto = 1
       try:
           totB = int(totB)
       except:
           totB = 0

       try:
            
           if check==0:
                label = 0

 
           elif check==1:
                label = 1
                #Training Dataset


           xdata.append([float(dur), tempProto, Sip, Dip, totB, totB/float(dur)])
           #xdata.append([Sip, Dip])
           ydata.append(label)
           count=count+1
       except Exception as e:
           #print(e)
           continue




    file = open('../dataset/test.csv', 'r')
    file.readline()
    for line in file:
        sd = line[:-1].split(',')
        check = 0
        sayac = 0
        while sayac < len(sd):
            sd[sayac] = sd[sayac].replace("\"", "")
            sayac = sayac + 1
        dur, proto, Sip, Dip, totB, label = sd[1], sd[4], sd[2], sd[3], sd[-2], sd[-1]

        sayac = 0
        while sayac < len(liste):
            if Sip == liste[sayac]:
                check = 1
                break
            elif Dip == liste[sayac]:
                check = 1
                break
                

            sayac = sayac + 1
        # back, nor, bot
        try:
            Sip = socket.inet_aton(Sip)
            Sip = struct.unpack("!L", Sip)[0]
        except:
            continue
        try:
            Dip = socket.inet_aton(Dip)
            Dip = struct.unpack("!L", Dip)[0]
        except:
            continue

        try:
            tempProto = protoDict[proto]
        except:
            tempProto = 1
        try:
            totB = int(totB)
        except:
            totB = 0

        try:

            if check == 0:
                label = 0


            elif check == 1:
                label = 1

            xdataT.append([float(dur), tempProto, Sip, Dip, totB, totB/float(dur)])
            #xdataT.append([Sip, Dip])
            ydataT.append(label)
            count=count+1


                
        except Exception as e:

            continue

    #pickle the dataset for fast loading
    print("pickle yukleniyor")
    file = open('../dataset/flowdata.pickle', 'wb')
    pickle.dump([np.array(xdata), np.array(ydata)], file)
    print("pickletest yukleniyor")
    file = open('../dataset/flowdatatest.pickle', 'wb')
    pickle.dump([np.array(xdataT), np.array(ydataT)], file)

    #return the training and the test dataset
    return np.array(xdata), np.array(ydata), np.array(xdataT), np.array(ydataT)

if __name__ == "__main__":
    loaddata('../dataset/training.csv')
