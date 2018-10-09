###############
# DESCRIPTION #
###############

# Python code for task environment used for meta-RL.


#############
# LIBRARIES #
#############
import numpy as np
import moviepy.editor as mpy
from PIL import Image
from PIL import ImageDraw 
from PIL import ImageFont


###################################
# ENVIRONMENT OF TWO-ARMED BANDIT #
###################################

class Two_Armed_Bandit():
    def __init__(self,difficulty):
        self.n_actions = 2
        self.difficulty = difficulty
        #self.reset()
        
    def set_restless_prob(self):    # sample from random walk list
        self.bandit = np.array([self.restless_list[self.timestep],1 - self.restless_list[self.timestep]])  
        
    def reset(self):
        self.timestep = 0
        if self.difficulty == "restless":       # bandit probability random-walks within an episode
            variance = np.random.uniform(0,.5)  # degree of random walk
            self.restless_list = np.cumsum(np.random.uniform(-variance,variance,(150,1)))   # calculation of random walk
            self.restless_list = (self.restless_list - np.min(self.restless_list)) / (np.max(self.restless_list - np.min(self.restless_list))) 
            self.set_restless_prob()
        if self.difficulty == "easy": bandit_prob = np.random.choice([0.9,0.1])
        if self.difficulty == "medium": bandit_prob = np.random.choice([0.75,0.25])
        if self.difficulty == "hard": bandit_prob = np.random.choice([0.6,0.4])
        if self.difficulty == "uniform": bandit_prob = np.random.uniform()
        if self.difficulty == "independent": self.bandit = np.random.uniform(size=2)

        if self.difficulty != "restless" and self.difficulty != "independent":
            self.bandit = np.array([bandit_prob,1 - bandit_prob])

        return self.bandit
        
    def step(self,action):
        #Get a random number.
        if self.difficulty == "restless": self.set_restless_prob()  # sample from random walk list
        self.timestep += 1
        bandit = self.bandit[action]
        result = np.random.uniform()
        if result < bandit:
            #return a positive reward.
            reward = 1
        else:
            #return a negative reward.
            reward = 0
        if self.timestep > 99: 
            done = True
        else: 
            done = False
        return reward,done,self.timestep
    
    def make_gif(self,buffer,path,count):     
        font = ImageFont.truetype("./resources/FreeSans.ttf", 24)
        images=[]
        r_cumulative=[0,0]
        for i in range(len(buffer)):
            r_cumulative[int(buffer[i,1])]+=buffer[i,2]
            bandit_image = Image.open('./resources/bandit.png')
            draw = ImageDraw.Draw(bandit_image)
            draw.text((40, 10),str(float("{0:.2f}".format(self.bandit[0]))),(0,0,0),font=font)
            draw.text((130, 10),str(float("{0:.2f}".format(self.bandit[1]))),(0,0,0),font=font)
            draw.text((60, 370),'Trial: ' + str(int(buffer[i,0])),(0,0,0),font=font)
            bandit_image = np.array(bandit_image)
            bandit_image[115:115+math.floor(r_cumulative[0]*2.5),20:75,:] = [0,255.0,0] 
            bandit_image[115:115+math.floor(r_cumulative[1]*2.5),120:175,:] = [0,255.0,0]
            bandit_image[101:107,10+(int(buffer[i,1])*95):10+(int(buffer[i,1])*95)+80,:] = [80.0,80.0,225.0]
            images.append(bandit_image)
        images=np.array(images)
        filename=path+'/'+str(count)+'.gif'
        duration=len(images)*0.1
        def make_frame(t):
            try:
                x = images[int(len(images)/duration*t)]
            except:
                x = images[-1]
            return x.astype(np.uint8)
        clip = mpy.VideoClip(make_frame, duration=duration)
        clip.write_gif(filename, fps = len(images) / duration, verbose=False, progress_bar=False)