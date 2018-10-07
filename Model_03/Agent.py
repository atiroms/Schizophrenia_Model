###############
# DESCRIPTION #
###############


#############
# LIBRARIES #
#############
import numpy as np
import tensorflow as tf
import scipy.signal
import time
import pandas as pd
from Network import *


#############
# A2C AGENT #
#############

class A2C_Agent():
    def __init__(self,id,param,environment,trainer,saver,global_episodes):
        self.id = id
        self.name = "agent_" + str(id)
        self.param=param
        self.env = environment
        self.n_actions=self.env.n_actions
        self.trainer = trainer
        self.saver=saver
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        # store summaries for each agent
        self.summary_writer = tf.summary.FileWriter(self.param.path_save+"/summary/"+self.name)

        # Create the local copy of the network and the tensorflow op to copy master paramters to local network
        self.local_AC = LSTM_RNN_Network(self.param,self.n_actions,self.name,trainer)
        #self.update_local_ops = update_target_graph('master',self.name)

    # Used to set worker network parameters to those of global network.
    def update_target_graph(self,from_scope,to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    # Discounting function used to calculate discounted returns.
    def discount(self,x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
        
    def train(self,episode_buffer,sess):
        timesteps = episode_buffer[:,0]
        actions = episode_buffer[:,1]
        rewards = episode_buffer[:,2]
        values = episode_buffer[:,3]
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        
        self.pr = prev_rewards
        self.pa = prev_actions
        # Here we take the rewards and values from the episode_buffer, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [self.param.bootstrap_value])
        discounted_rewards = self.discount(self.rewards_plus,self.param.gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [self.param.bootstrap_value])
        advantages = rewards + self.param.gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = self.discount(advantages,self.param.gamma)

        rnn_state = self.local_AC.state_init
        feed_dict = {
            self.local_AC.target_v:discounted_rewards,
            self.local_AC.prev_rewards:np.vstack(prev_rewards),
            self.local_AC.prev_actions:prev_actions,
            self.local_AC.actions:actions,
            self.local_AC.timestep:np.vstack(timesteps),
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:rnn_state[0],
            self.local_AC.state_in[1]:rnn_state[1]}
        t_l,v_l,p_l,e_l,g_n,v_n,_ = sess.run([
            self.local_AC.loss,
            self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        t_l /= episode_buffer.shape[0]
        v_l /= episode_buffer.shape[0]
        p_l /= episode_buffer.shape[0]
        e_l /= episode_buffer.shape[0] 

        return t_l, v_l, p_l, e_l, g_n, v_n
        
    def work(self,sess,coord):
        episode_count_global = sess.run(self.global_episodes)           # refer to global episode counter over all agents
        episode_count_local = 0
        agent_steps = 0
        #print("Starting " + self.name + "                    ")
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():                              # iterate over episodes
                episode_count_global = sess.run(self.global_episodes)   # refer to global episode counter over all agents
                sess.run(self.increment)                                # add to global episode counter
                print("Running global episode: " + str(episode_count_global) + ", " + self.name + " local episode: " + str(episode_count_local)+ "          ", end="\r")
                t_start = time.time()
                sess.run(self.update_target_graph(self.name,'main'))                        # copy master graph to local
                episode_buffer = []
                episode_values = []
                #episode_frames = []
                episode_reward = [0,0]
                episode_steps = 0                                       # counter of steps within an episode
                d = False
                r = 0
                a = 0
                t = 0
                bandit = self.env.reset()                               # returns np.array of bandit probabilities
                rnn_state = self.local_AC.state_init                    # returns zero array with LSTM cell size
                
                # act
                while d == False:                                       # d is "done" flag returned from the environment
                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state_new = sess.run(
                        [self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                        feed_dict={
                            self.local_AC.prev_rewards:[[r]],
                            self.local_AC.timestep:[[t]],
                            self.local_AC.prev_actions:[a],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    
                    rnn_state = rnn_state_new
                    r,d,t = self.env.step(a)                        
                    #episode_buffer.append([a,r,t,d,v[0,0]])
                    episode_buffer.append([t,a,r,v[0,0]])
                    episode_values.append(v[0,0])
                    #episode_frames.append(set_image_bandit(episode_reward,bandit,a,t))
                    episode_reward[a] += r
                    agent_steps += 1
                    episode_steps += 1

                episode_buffer=np.array(episode_buffer)
                
                # train the network using the experience buffer at the end of the episode.
                #if len(episode_buffer) != 0 and train == True:
                if self.param.train == True:
                    t_l,v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess)

                # Save activity in /activity/activity.h5 file
                df_episode = pd.DataFrame(episode_buffer)
                df_episode.columns = ['timestep', 'action', 'reward', 'value']
                df_episode.insert(loc=1, column='arm0_prob', value=bandit[0])
                df_episode.insert(loc=2, column='arm1_prob', value=bandit[1])
                df_episode.insert(loc=0, column='agent', value=self.id)
                df_episode.insert(loc=0, column='episode_count', value=episode_count_global)
                df_episode.ix[:,['episode_count','agent','timestep','action']]=df_episode.ix[:,['episode_count','agent','timestep','action']].astype('int64')

                hdf=pd.HDFStore(self.param.path_save+'/activity/activity.h5')
                hdf.put('activity',df_episode,format='table',append=True,data_columns=True)
                hdf.close()
                    
                # save model parameters as tensorflow saver
                if self.param.interval_ckpt>0:
                    if episode_count_global % self.param.interval_ckpt == 0 and self.param.train == True:
                        self.saver.save(sess,self.param.path_save+'/model/'+str(episode_count_global)+'.ckpt')
                        #print('Saved model parameters at global episode ' + str(episode_count_global) + '.                 ')

                # save model trainable variables in hdf5 format
                if self.param.interval_var>0:
                    if episode_count_global % self.param.interval_var == 0 and self.param.train == True:
                        master_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master')
                        val = sess.run(master_vars)                      
                        all_vars=np.empty(shape=[0,])
                        for v in val:
                            all_vars=np.concatenate((all_vars,v.ravel()),axis=0)
                        all_vars=pd.DataFrame(all_vars,columns=['value'])
                        all_vars.insert(loc=0, column='variable', value=range(all_vars.shape[0]))
                        all_vars.insert(loc=0, column='episode_count',value=episode_count_global)
                        #all_vars=all_vars.reshape((1,-1))
                        #all_vars=pd.DataFrame(all_vars,columns=('var'+str(i) for i in range(all_vars.shape[1])))
                        #print(type(all_vars))
                        hdf=pd.HDFStore(self.param.path_save+'/model/variable.h5')
                        hdf.put('variable',all_vars,format='table' ,append=True,data_columns=True)
                        hdf.close()

                # save gif image of fast learning
                if self.param.interval_pic>0:
                    if episode_count_global % self.param.interval_pic == 0:
                        self.env.make_gif(episode_buffer, self.param.path_save + '/pic', episode_count_global)

                # Save episode summary in /summary folder
                summary_episode = tf.Summary()
                summary_episode.value.add(tag="Performance/Reward", simple_value=float(np.sum(episode_reward)))
                summary_episode.value.add(tag="Performance/Mean State-Action Value", simple_value=float(np.mean(episode_values)))
                summary_episode.value.add(tag="Simulation/Calculation Time", simple_value=float(time.time()-t_start))
                summary_episode.value.add(tag="Environment/Step Length", simple_value=int(episode_steps))
                summary_episode.value.add(tag="Environment/Arm0 Probability", simple_value=float(bandit[0]))
                summary_episode.value.add(tag="Environment/Arm1 Probability", simple_value=float(bandit[1]))
                if self.param.train == True:
                    summary_episode.value.add(tag="Loss/Total Loss", simple_value=float(t_l))
                    summary_episode.value.add(tag="Loss/Value Loss", simple_value=float(v_l))
                    summary_episode.value.add(tag="Loss/Policy Loss", simple_value=float(p_l))
                    summary_episode.value.add(tag="Loss/Policy Entropy", simple_value=float(e_l))
                    summary_episode.value.add(tag="Loss/Gradient L2Norm", simple_value=float(g_n))
                    summary_episode.value.add(tag="Loss/Variable L2Norm", simple_value=float(v_n))
                self.summary_writer.add_summary(summary_episode, episode_count_global)
                self.summary_writer.flush()

                if episode_count_global == self.param.episode_stop:
                    print('Reached maximum episode count: '+ str(episode_count_global) + '.                           ')
                    break

                episode_count_local += 1        # add to local counter in all agents

