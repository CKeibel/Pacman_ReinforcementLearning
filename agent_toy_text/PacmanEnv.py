import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

import socket as skt
import threading
import json
from builtins import print
from struct import unpack

socket = skt.socket(skt.AF_INET, skt.SOCK_STREAM)
buffer = []


def hasNext():
    return len(buffer) > 0


def nextString():
    return buffer.pop(0)


class BGThread(threading.Thread):
    def run(self):
        print('Client started\n')
        try:
            socket.connect(('localhost', 6000))
        except skt.error:
            socket.close()
            print('connecting failed')
            return
        print('connected')
        while True:
            try:
                threading.Event().wait(0.001)

                data = read()
                if data == 'SERVER_STOPPED':
                    break
                if not (str.startswith(data, 'PREPARING_GAME')):
                    buffer.append(data)
            except Exception as e:
                print('terminated BG-Thread')
                raise e
        print('terminated BG-Thread')


def read():
    size = 0
    try:
        data = socket.recv(1)
        if len(data) == 0:
            raise Exception("Connection lost")
        # unpack the received data from bytes to integer
        packages = unpack('b', data)[0]
        # return the packages to char and cast it to int
        packages = int(chr(packages))
        # print("packages: "+str(packages))

        # print("packages "+str(packages))
        completeData = ""
        for i in range(0, packages):
            data = socket.recv(1)
            # unpack the received data from bytes to integer
            size = unpack('b', data)[0]
            # return the size to char and cast it to int to be used for pow
            size = int(chr(size))
            size = pow(2, size)

            data = socket.recv(size).decode('UTF-8')

            rawData = repr(data)
            if i < packages - 1:
                procData = rawData[1:len(rawData) - 1]  # without the leading and trailing '
            else:
                procData = ""
                i = 0
                # print("rawData: "+str(rawData))
                while True:
                    procData += rawData[i]
                    # print(procData)
                    i = i + 1
                    if procData.endswith('\\x00'):
                        procData = procData[0:len(procData) - 4]
                        break
                    if i == len(rawData):
                        break
                procData = procData[1:len(procData)]  # without the leading '

            completeData += procData

        return completeData
    except OverflowError as e:
        print(e)
        print('Received ' + str(size) + ' bytes')
        return ''


def write(data):
    paddedLength = findBest2power(len(data))

    socket.sendall(str(paddedLength).encode('UTF-8'))

    paddedLength = pow(2, paddedLength)

    paddedBytesToWrite = str('')
    for b in range(0, paddedLength):
        paddedBytesToWrite = paddedBytesToWrite + ' '

    paddedBytesToWrite = data + paddedBytesToWrite[len(data): paddedLength]

    socket.sendall(str(paddedBytesToWrite).encode('UTF-8'))


def findBest2power(length):
    i = 7
    while length >= pow(2, i):
        i += 1
    return i


def writeAgentConfig():
    name = "MyPythonAgent"
    clazz = "de.fh.pacman.Pacman"
    agentConfig = "{\n\"class\":AgentConfig\n\"name\":" + name + "\n\"entityClass\":" + clazz + "\n}"
    write(agentConfig)


class PacmanEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    
    self.action_space = spaces.Discrete(4)#up, left, right, down
    
    # Observations PACMAN, HUNTER, RANDOM, EAGER, PASSIVE, DOTS, ENV_SIZE
    self.low = np.array([7, 0, 0, 0, 0, 0], dtype=np.int8)
    self.high = np.array([19, 19, 19, 19, 19, 8], dtype=np.int8)

    self.observation_space = spaces.box.Box(low=self.low, high=self.high, dtype=np.int8)

    self.actionEffect = ""
    self.percept = None
    self.totalDots = -1
    self.reward_dots = 8
    self.tmp_dots = 8
    self.steps = 0

    
    t = BGThread()
    t.start()
    threading.Event().wait(0.01)

  def countDots(self, view):
      dots = 0
      for x in range(0, len(view)):
          for y in range(0, len(view[x])):
              t = view[x][y]
              if view[x][y] == "D":
                  dots = dots+1
      return dots

  def step(self, action):
    # reward
    self.steps += 1
    if self.percept is None:
        reward = 0
    elif self.steps > 4:
        self.tmp_dots = self.countDots(self.percept['view'])
       # print("Vorher:", self.reward_dots, "Aktuell:", self.tmp_dots)
        if self.reward_dots > self.tmp_dots:
            
            reward = 1 #/ self.steps
            self.reward_dots = self.tmp_dots
        else:
            reward = 0
    else: 
        reward = 0
        #tmp_dots = self.countDots(self.percept['view']) / self.steps*100
        #reward = self.totalDots - tmp_dots
        #print("TotalDots:", self.totalDots)

        

    # observation 
    ob = np.array([7, 0, 0, 0, 0, 7])  
    if self.percept is None:
        ob = np.array([7, 0, 0, 0, 0, 7])
    else:
        # Pacman Position
        pacman_pos = self.percept['position']
        pacman_pos = self.get_position(pacman_pos[0], pacman_pos[1])
        ob[0] = pacman_pos

        # Ghosts Position
        ghosts = self.percept['ghostTypes']
        for pos, ghost in ghosts.items():
            x, y = self.index_to_ints(pos)
            
            ghost_pos = self.get_position(x, y)

            if ghost == 'GHOST_HUNTER':
                ob[1] = ghost_pos
            elif ghost == 'GHOST_RANDOM':
                ob[2] = ghost_pos
            elif ghost == 'GHOST_EAGER':
                ob[3] = ghost_pos
            elif ghost == 'GHOST_PASSIVE':
                ob[4] = ghost_pos

        dots = self.countDots(self.percept['view'])
        ob[5] = dots


        
    
    while not hasNext():
        threading.Event().wait(0.001)
    data = nextString()
    js = json.loads(data, encoding="UTF-8")
    # print("js: "+str(js))

    # Differ between received serverSignals, percepts and actionEffects
    done = False
    if isinstance(js, list):
        pass
    elif isinstance(js, dict):
        if js["class"] == "PacmanActionEffect":
            pass
        elif js["class"] == "PacmanPercept":
            self.percept = js
            if self.totalDots == -1:
                self.totalDots = self.countDots(js["view"])
            write('\"'+action+'\"')
        elif js["class"] == "PacmanGameResult":
            done = True
            pass
    
    return ob, reward, done, {}
  
  def reset(self):
    self.reward_dots = self.tmp_dots = 8
    self.steps = 0

    if self.actionEffect == "" :
        writeAgentConfig()
        self.actionEffect = "GAME_INITIALIZED"
        while True:
            while not hasNext():
                threading.Event().wait(0.001)
            data = nextString()
            if data == '["STARTING_GAME"]':
                break
    print("Resetted")
    
    return np.array([7, 0, 19, 0, 0, 7]) ################################################################ Inital State je nach Map anpassen
  
  def render(self, mode='human'):
    ...
  def close(self):
    ...

  def index_to_ints(self, index):
    str1 = index
    x = None
    y = None

    try:
        x = int(str1[1] + str1[2])
    except:
        try: 
            x = int(str1[1])
        except:
            x = 1

    try: 
        y = int(str1[5] + str1[6])
    except:
        try:
            y = int(str1[4] + str1[5])   
        except:
            try:
                y = int(str1[4])
            except:
                try:
                    y = int(str1[5])
                except:
                    y = 1


    return x, y

  def get_position(self, x, y):
    position = (len(self.percept['view']) * y) + (x + 1)

    return position
