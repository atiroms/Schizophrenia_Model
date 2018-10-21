###############
# DESCRIPTION #
###############

# Python code for reinforcement learning agents used for meta-RL.


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
    def __init__(self,id,param,environment,trainer,saver,episode_global):
        self.id = id
        self.name = "agent_" + str(id)
        self.param=param
        self.env = environment
        self.n_actions=self.env.n_actions
        self.trainer = trainer
        self.saver=saver
        self.episode_global = episode_global
        self.increment = self.episode_global.assign_add(1)

        # store summaries for each agent
        #self.summary_writer = tf.summary.FileWriter(self.param.path_save+"/summary/"+self.name)

        # Create the local copy of the network and the tensorflow op to copy master paramters to local network
        self.local_AC = LSTM_RNN_Network(self.param,self.n_actions,self.name,trainer)
        #self.update_local_ops = update_target_graph('master',self.name)

        self.init_df()

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

    def init_df(self):
        if self.param.train == True:
            self.df_summary=pd.DataFrame(columns=['episode','reward','value','step_episode',
                                                  'prob_arm0','prob_arm1','time_calc',
                                                  'loss_total','loss_value','loss_policy',
                                                  'loss_entropy','norm_gradient','norm_variable'])
            self.df_variable=pd.DataFrame(columns=['episode','variable','value'])
            for col in ['episode','variable']:
                self.df_variable.loc[:,col]=self.df_variable.loc[:,col].astype('int64')
        else:
            self.df_summary=pd.DataFrame(columns=['episode','reward','value','step_episode',
                                                  'prob_arm0','prob_arm1','time_calc'])
        for col in ['episode','step_episode']:
            self.df_summary.loc[:,col]=self.df_summary.loc[:,col].astype('int64')
        for col in ['reward']:
            self.df_summary.loc[:,col]=self.df_summary.loc[:,col].astype('float64')
        self.df_activity=pd.DataFrame(columns=['episode','id_agent','prob_arm0','prob_arm1',
                                               'timestep','action','reward','value'])
        for col in ['episode','action','id_agent','timestep']:
            self.df_activity.loc[:,col]=self.df_activity.loc[:,col].astype('int64')
 
    def train(self,episode_buffer,sess):
        timesteps = episode_buffer[:,0]
        actions = episode_buffer[:,1]
        rewards = episode_buffer[:,2]
        values = episode_buffer[:,3]
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        
        self.pr = prev_rewards
        self.pa = prev_actions
        # The advantage function according to "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [self.param.bootstrap_value])
        discounted_rewards = self.discount(self.rewards_plus,self.param.gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [self.param.bootstrap_value])
        advantages = rewards + self.param.gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = self.discount(advantages,self.param.gamma)

        rnn_state = self.local_AC.state_init    # array of zeros defined in Network
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
        cnt_episode_global = sess.run(self.episode_global)           # refer to global episode counter over all agents
        cnt_episode_local = 0
        agent_steps = 0
        #print("Starting " + self.name + "                    ")
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():                              # iterate over episodes
                cnt_episode_global = sess.run(self.episode_global)   # refer to global episode counter over all agents
                sess.run(self.increment)                                # add to global episode counter
                print("Running global episode: " + str(cnt_episode_global) + ", " + self.name + " local episode: " + str(cnt_episode_local)+ "          ", end="\r")
                t_start = time.time()
                sess.run(self.update_target_graph('master',self.name))                        # copy master graph to local
                episode_buffer = []
                episode_values = []
                #episode_frames = []
                episode_reward = [0,0]
                step_episode = 0                                       # counter of steps within an episode
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
                    step_episode += 1

                episode_buffer=np.array(episode_buffer)
                
                # train the network using the experience buffer at the end of the episode.
                if self.param.train == True:
                    t_l,v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess)

                # Save simulation summary in dataframe
                if self.param.interval_summary>0:
                    if cnt_episode_global % self.param.interval_summary == 0:
                        df_summary_episode=pd.DataFrame(data=[[cnt_episode_global,np.sum(episode_reward),
                                                              np.mean(episode_values),step_episode,
                                                              bandit[0],bandit[1],time.time()-t_start]],
                                                        columns=['episode','reward','value',
                                                                 'step_episode','prob_arm0','prob_arm1',
                                                                 'time_calc'])
                        if self.param.train == True:
                            df_summary_episode=df_summary_episode.assign(loss_total=t_l,loss_value=v_l,
                                                                         loss_policy=p_l,loss_entropy=e_l,
                                                                         norm_gradient=g_n,norm_variable=v_n)
                        for col in ['episode','step_episode']:
                            df_summary_episode.loc[:,col]=df_summary_episode.loc[:,col].astype('int64')
                        
                        self.df_summary=self.df_summary.append(df_summary_episode)

                # Save model trainable variables in dataframe
                if self.param.interval_var>0:
                    if cnt_episode_global % self.param.interval_var == 0 and self.param.train == True:
                        master_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master')
                        val = sess.run(master_vars)                      
                        df_var_episode=np.empty(shape=[0,])
                        for v in val:
                            df_var_episode=np.concatenate((df_var_episode,v.ravel()),axis=0)
                        df_var_episode=pd.DataFrame(df_var_episode,columns=['value'])
                        df_var_episode.insert(loc=0, column='variable', value=range(df_var_episode.shape[0]))
                        df_var_episode.insert(loc=0, column='episode',value=cnt_episode_global)
                        self.df_variable=pd.concat([self.df_variable,df_var_episode])

                # Save activity in dataframe
                if self.param.interval_activity>0:
                    if cnt_episode_global % self.param.interval_activity == 0:
                        df_activity_episode = pd.DataFrame(episode_buffer,columns=['timestep', 'action', 'reward', 'value'])
                        df_activity_episode=df_activity_episode.assign(episode=cnt_episode_global,id_agent=self.id,
                                                                       prob_arm0=bandit[0],prob_arm1=bandit[1])
                        for col in ['episode','action','id_agent','timestep']:
                            df_activity_episode.loc[:,col]=df_activity_episode.loc[:,col].astype('int64')
                        self.df_activity=pd.concat([self.df_activity,df_activity_episode])

                # Sersisitent saving of model summary, parameters and activity
                if self.param.interval_persist>0:
                    if cnt_episode_global>0 and cnt_episode_global % self.param.interval_persist == 0:
                        hdf=pd.HDFStore(self.param.path_save+'/summary/summary.h5')
                        hdf.put('summary',self.df_summary,format='table',append=True,data_columns=True)
                        hdf.close()

                        hdf=pd.HDFStore(self.param.path_save+'/model/variable.h5')
                        hdf.put('variable',self.df_variable,format='table',append=True,data_columns=True)
                        hdf.close()

                        hdf=pd.HDFStore(self.param.path_save+'/activity/activity.h5')
                        hdf.put('activity',self.df_activity,format='table',append=True,data_columns=True)
                        hdf.close()

                        self.init_df()
                        
                # save gif image of fast learning
                if self.param.interval_pic>0:
                    if cnt_episode_global % self.param.interval_pic == 0:
                        self.env.make_gif(episode_buffer, self.param.path_save + '/pic', cnt_episode_global)

                # save model parameters as tensorflow saver
                if self.param.interval_ckpt>0:
                    if cnt_episode_global % self.param.interval_ckpt == 0 and self.param.train == True:
                        self.saver.save(sess,self.param.path_save+'/model/'+str(cnt_episode_global)+'.ckpt')
                        #print('Saved model parameters at global episode ' + str(cnt_episode_global) + '.                 ')

                # Save episode summary in /summary folder
                '''
                summary_episode = tf.Summary()
                summary_episode.value.add(tag="Performance/Reward", simple_value=float(np.sum(episode_reward)))
                summary_episode.value.add(tag="Performance/Mean State-Action Value", simple_value=float(np.mean(episode_values)))
                summary_episode.value.add(tag="Simulation/Calculation Time", simple_value=float(time.time()-t_start))
                summary_episode.value.add(tag="Environment/Step Length", simple_value=int(step_episode))
                summary_episode.value.add(tag="Environment/Arm0 Probability", simple_value=float(bandit[0]))
                summary_episode.value.add(tag="Environment/Arm1 Probability", simple_value=float(bandit[1]))
                if self.param.train == True:
                    summary_episode.value.add(tag="Loss/Total Loss", simple_value=float(t_l))
                    summary_episode.value.add(tag="Loss/Value Loss", simple_value=float(v_l))
                    summary_episode.value.add(tag="Loss/Policy Loss", simple_value=float(p_l))
                    summary_episode.value.add(tag="Loss/Policy Entropy", simple_value=float(e_l))
                    summary_episode.value.add(tag="Loss/Gradient L2Norm", simple_value=float(g_n))
                    summary_episode.value.add(tag="Loss/Variable L2Norm", simple_value=float(v_n))
                self.summary_writer.add_summary(summary_episode, cnt_episode_global)
                self.summary_writer.flush()
                '''
                if cnt_episode_global == self.param.episode_stop:
                    print('Reached maximum episode count: '+ str(cnt_episode_global) + '.                           ')
                    break

                cnt_episode_local += 1        # add to local counter in all agents