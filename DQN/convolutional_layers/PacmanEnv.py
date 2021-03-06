import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

import socket as skt
import threading
import json
from builtins import print
from struct import unpack
import collections

from PIL import Image, ImageDraw


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
    #Framestack
    self.buffer_frames = 2
    self.frames = collections.deque(maxlen=self.buffer_frames)
    #self.frame_stack = np.zeros(shape=(1, 84, 84, 3*self.buffer_frames), dtype=np.float32)
    
    self.action_space = spaces.Discrete(4)#up, left, right, down
    self.observation_space = spaces.box.Box(low=16, high=255, shape=(84, 84, 1), dtype=np.uint8)
    low = np.repeat(self.observation_space.low[np.newaxis, ...], self.buffer_frames, axis=0)
    high = np.repeat(self.observation_space.high[np.newaxis, ...], self.buffer_frames, axis=0)
    self.observation_space = gym.spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)

    self.actionEffect = ""
    self.percept = None
    self.totalDots = -1
    self.reward_dots = 19
    self.tmp_dots = 19
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
    
    # if self.percept is None:
    #     reward = 0
    # elif self.steps > 4:
    #     self.tmp_dots = self.countDots(self.percept['view'])
    #     if self.reward_dots > self.tmp_dots:   
    #         reward = 1000
    #         if self.tmp_dots <= 1:
    #             reward = 1000
    #         self.reward_dots = self.tmp_dots
    #     else:
    #         reward = -5
    # else: 
    #     reward = 0
    if self.percept is None:
        reward = 0
    else:
        if self.steps > 3:
            self.tmp_dots = self.countDots(self.percept['view'])
            if self.reward_dots > self.tmp_dots:
                reward = (self.reward_dots - self.tmp_dots)*5 
                self.reward_dots = self.tmp_dots
            else:
                reward = 0
        else:
            reward = 0

        


        

    # observation
    if self.percept is None:
        ob = self.initial_state()
    else:
        ob = self.percept['view']

    ob = self.draw_state(ob)
    # buffer Frames
    self.frames.append(ob)
    frame_stack = np.asarray(self.frames, dtype=np.float32)
    frame_stack = np.moveaxis(frame_stack, 0, -1).reshape(1, 84, 84, -1)

        
    
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

    
    return frame_stack, reward, done, {}
  
  def reset(self):
    # resseting
    self.reward_dots = self.tmp_dots = 19
    self.steps = 0
    ob = self.initial_state()
    ob = self.draw_state(ob)
    self.frames = collections.deque(maxlen=self.buffer_frames)
    frame_stack = np.zeros(shape=(1, 84, 84, 1*self.buffer_frames), dtype=np.float32)
    #print(frame_stack.shape)
    
    for i in range(self.buffer_frames):
        self.frames.append(ob)


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
    
    return frame_stack 


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

  def draw_state(self, state):
    im = Image.new("RGB", (len(state)*12, len(state[0])*12))
    draw = ImageDraw.Draw(im)

    for x in range(0, len(state)):
        for y in range(0, len(state[x])):
            if state[y][x] == 'W':
                draw.rectangle((x*12, y*12, x*12+12, y*12+12), fill=(0, 0, 139))
            elif state[y][x] == 'P':
                draw.rectangle((x*12, y*12, x*12+12, y*12+12), fill=(255, 255 ,0))
            elif state[y][x] == 'G' or state[x][y] == 'GD':
                draw.rectangle((x*12, y*12, x*12+12, y*12+12), fill=(255, 0, 0))
            elif state[y][x] == 'D':
                draw.rectangle((x*12, y*12, x*12+12, y*12+12), fill=(255, 255, 255))
            elif state[y][x] == 'E':
                draw.rectangle((x*12, y*12, x*12+12, y*12+12), fill=(0, 204, 0))
    #im.show()
    im = im.convert('L')
    im = np.array(im)
    im = np.reshape(im, (1, 84, 84, 1))
        
    return im

  def initial_state(self):
    state = [['W' for _ in range(7)],
            ['W', 'P', 'D', 'D', 'D', 'D', 'W'],
            ['W', 'D', 'W', 'D', 'W', 'D', 'W'],
            ['W', 'D', 'D', 'D', 'D', 'D', 'W'],
            ['W', 'D', 'W', 'D', 'W', 'D', 'W'],
            ['W', 'D', 'D', 'D', 'D', 'G', 'W'],
            ['W' for _ in range(7)]]
    return state



