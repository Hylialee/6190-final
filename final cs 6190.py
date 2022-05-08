# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 22:48:00 2022

@author: Hylia
"""
import pyautogui as grab
import time
import numpy as np

#these are the pixels the algorithm searches to see whether a match has started or ended
startpixels = ((771, 405), (429, 540), (421, 745), (444, 452), (366, 277), (531, 317), (1320, 416), (1300, 639), (953, 285), (906, 599), (1102, 711), (1404, 674), (379, 303), (1025, 184), (888, 693), (1129, 698), (440, 434), (317, 226), (1539, 216), (1606, 855), (1240, 724))  
endpixels = ((390, 748), (464, 485), (536, 638), (635, 754), (904, 515), (972, 535), (1176, 491), (1325, 556), (1365, 380), (1390, 302), (1338, 567), (1389, 297), (1332, 577), (1441, 617), (1595, 490))
antipixel = ((1549, 364), (1672, 702))
#these are the collors the RISC bar can take
RISCbcol = (122, 122, 122)
RISCpcol = (210, 13, 224)
RISCrcol = (234, 27, 11)
RISCycol = (242, 190, 2)
#these are important coordinates  for searching the risc bar
RISCp1full = ((621, 183), (637, 183), (635, 188), (635, 198), (637, 193), (637, 203), (637, 213)) #see if RISC is full p1
RISCp1check = ((649, 187), (657, 187), (665, 187), (678, 188), (710, 187), (720, 187), (734, 188), (753, 187), (867, 187)) #points for if someone is in front of the RISC bar p1
RISCp1 = np.zeros((20, 2)) #creating the checks to see how full the bar is
for i in range(0, 20):
    RISCp1[i][0] = 855 - 11 * i
    RISCp1[i][1] = 191
#same important coord types, but for Player 2
RISCp2 = np.zeros((20, 2))
RISCp2check = ((1074, 191), (1093, 189), (1107, 190), (1120, 190), (1168, 190), (1180, 188), (1207, 190), (1223, 190), (1243, 190), (1254, 190), (1264, 190), (1273, 190))
RISCp1full = ((1282, 202), (1298, 202), (1300, 189), (1300, 179), (1286, 179), (1284, 187))
for i in range(0, 20):
    RISCp2[i][0] = 1057 + 11 * i
    RISCp2[i][1] = 191
#for tracking health
colorsp1 = [[[242.,229.,191.],
  [204.,180.,156.],
  [204.,180.,158.],
  [183.,157.,138.],
  [181.,154.,135.],
  [183.,158.,139.],
  [178.,153.,135.],
  [175.,151.,132.],
  [214.,191.,167.],
  [191.,167.,145.],
  [177.,153.,131.],
  [173.,150.,129.],
  [193.,174.,151.],
  [157.,142.,124.],
  [ 17.,128.,204.],
  [ 16.,128.,204.],
  [ 16.,141.,223.],
  [ 24.,153.,227.],
  [ 65.,196.,239.],
  [ 31.,167.,232.],
  [235.,109.,113.]],

 [[  0.,  0.,  0.],
  [243.,223.,186.],
  [243.,219.,183.],
  [242.,215.,180.],
  [243.,212.,179.],
  [243.,210.,176.],
  [243.,207.,175.],
  [243.,205.,173.],
  [243.,201.,171.],
  [243.,200.,169.],
  [243.,197.,168.],
  [243.,194.,166.],
  [243.,192.,163.],
  [243.,188.,161.],
  [242.,183.,159.],
  [242.,179.,155.],
  [242.,173.,152.],
  [239.,169.,153.],
  [220.,170.,176.],
  [237.,158.,150.],
  [  0.,  0.,  0.]]]
healthalt = np.array((244, 153, 8))
#for generating new colors
'''screen = grab.screenshot()
pixels = screen.load()
colors = np.zeros((2, 21, 3))
colors[0][0] = pixels[151, 131]
for i in range(1, 20):
    for j in range(0, 2):
        colors[j, i] = pixels[150 + 36 * i, 157 - 18*j]
colors[0][20] = pixels[868, 126]
print(np.array2string(colors, separator = ","))'''

def baseloop(): #the basic loop for the entire algorithm
    secondcount = 0
    status = "start"
    RISCp1g = np.array([0.])
    for t in range(1, 10000):
        ms = time.time()*1000 % 4000
        screen = grab.screenshot()
        pixels = screen.load()
        count = 0
        #process to run in between rounds
        if(status == "end"):
            for i in startpixels:
                if(pixels[i[0],i[1]] == (255, 255, 255)):
                    count = count + 1
                else: 
                    break
            for i in antipixel:
                if(pixels[i[0], i[1]] == (255, 255, 255)):
                    break
                else:
                    count = count + 1
            if(count == len(startpixels) + len(antipixel)):
                print("Match Start")
                status = "start"
        #process to run during the round
        if(status == "start"):
            status = endloop(pixels) #check if round is over
            newRISCp1g = RISCtrack(pixels, RISCp1g[-1], 1)
            RISCp1g = np.append(RISCp1g, newRISCp1g) #find the new RISC value for player 1
        #test code area
        if(secondcount == 10):
            #print(len(RISCp1g))
            Healthp1 = 0.051
            fullhdist = np.asarray(pixels[151, 131]) - colorsp1[0][0]
            fullhdist = np.sqrt(fullhdist @ fullhdist)
            zerohdist = np.asarray(pixels[868, 126]) - colorsp1[0][20]
            zerohdist = np.sqrt(zerohdist @ zerohdist)
            zerohdist2 = np.asarray(pixels[868, 126]) - healthalt
            zerohdist2 = np.sqrt(zerohdist2 @ zerohdist2)
            zerohdist = min(zerohdist, zerohdist2)
            if(fullhdist < 20):
                Healthp1 = 1
            elif(zerohdist > 40):
                Healthp1 = 0
                print(pixels[868, 126])
            else:
                for i in range(1, 20):
                    healthdist1 = np.asarray(pixels[150 + 36 * i, 157]) - colorsp1[0][i]
                    healthdist1 = np.sqrt(healthdist1 @ healthdist1)
                    healthdist2 = np.asarray(pixels[150 + 36* i, 139]) - colorsp1[1][i]
                    healthdist2 = np.sqrt(healthdist2 @ healthdist2)
                    healthdist3 = np.asarray(pixels[150 + 36 * i, 157]) - healthalt
                    healthdist3 = np.sqrt(healthdist3 @ healthdist3)
                    healthdist4 = np.asarray(pixels[150 + 36 * i, 139]) - healthalt
                    healthdist4 = np.sqrt(healthdist4 @ healthdist4)
                    healthdist = min(healthdist1, healthdist2, healthdist3, healthdist4)
                    if(healthdist < 10):
                        Healthp1 = 1 - .05 * i
                        break
            print(Healthp1)
            secondcount = 0
        newms = time.time()*1000 % 4000
        time.sleep(max(49 - ((newms - ms) % 4000), 0)/1000)
        secondcount = secondcount + 1
        
def endloop(pixels): #checks to see if the match is ended
    count = 0
    status = "start"
    for i in endpixels:
        if(pixels[i[0], i[1]] == (255, 255, 255)):
            count = count + 1
        else: 
            break
    for i in antipixel:
        if(pixels[i[0], i[1]] == (255, 255, 255)):
            break
        else:
            count = count + 1
    if(count == len(endpixels) + len(antipixel)):
        print("Match End")
        status = "end"
    return(status)

def RISCtrack(pixels, lastrisc, player): #tells you how full the RISC gauge is
    if player == 1:
        RISCcheck = RISCp1check
        RISCcoord = RISCp1
        RISCfull = RISCp1full
    Rgaugep1 = 0
    blocked = 0
    #Checks if bar is full (with the little hazard light)
    for i in range(0, len(RISCp1full)):
        diff = np.asarray(pixels[RISCfull[i][0],RISCfull[i][1]]) - np.asarray(RISCycol)
        sqdiff = np.sqrt(diff @ diff)
        if sqdiff < 45:
            return(1)
            blocked = 1
    #Check if bar is blocked
    for i in range(0, len(RISCcheck)):
        if blocked == 1:
            break
        diff1 = np.asarray(pixels[RISCcheck[i][0],RISCcheck[i][1]]) - np.asarray(RISCbcol)
        diff2 = np.asarray(pixels[RISCcheck[i][0],RISCcheck[i][1]]) - np.asarray(RISCpcol)
        diff3 = np.asarray(pixels[RISCcheck[i][0],RISCcheck[i][1]]) - np.asarray(RISCrcol)
        sqdiff = min(np.sqrt(diff1 @ diff1), np.sqrt(diff2 @ diff2), np.sqrt(diff3 @ diff3))
        if sqdiff > 40:
            return(lastrisc)
            blocked = 1
            break
    #check how full bar is if it isn't blocked or full
    for i in range(0, 20):
        if blocked == 1:
            break
        diff1 = np.asarray(pixels[RISCcoord[i][0],RISCcoord[i][1]]) - np.asarray(RISCrcol)
        diff2 = np.asarray(pixels[RISCcoord[i][0],RISCcoord[i][1]]) - np.asarray(RISCpcol)
        sqdiff = min(np.sqrt(diff1 @ diff1), np.sqrt(diff2 @ diff2))
        if sqdiff < 100:
            Rgaugep1 = (i+1)/20
            #print(pixels[RISCp1[i][0],RISCp1[i][1]])
            #maxloc = RISCp1[i]
    if blocked != 1:
        return(Rgaugep1)
            
        
    
    