import gym
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Reshape, Concatenate, Lambda
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
from multiprocessing import Pool, Manager
import time
import os
from multiprocessing.queues import Empty

LOSS_CLIPPING=0.2
ENTROPY_LOSS=1e-3
DUMMY_VALUE, DUMMY_ACTION = np.zeros((1, 1)), np.zeros((1, 2, 4))
GAMMA=0.95

def proximal_policy_optimization_loss(advantage, old_prediction):#this is the clipped PPO loss function, see https://arxiv.org/pdf/1707.06347.pdf
    def loss(y_true, y_pred):#1 is the log of the std
        r = K.prod(K.exp(old_prediction[:,1])/(K.exp(y_pred[:,1])+K.epsilon())*(K.exp(((old_prediction[:,0]-y_true[:,0])**2)/(2*(K.exp(old_prediction[:,1])**2) + K.epsilon()) - ((y_pred[:,0]-y_true[:,0])**2)/(2*(K.exp(y_pred[:,1])**2) + K.epsilon()))), axis=1, keepdims=True)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * (K.sum(y_pred[:,1], axis=1, keepdims=True)))
    return loss

def create_model(weights=None):
    advantage = Input(shape=(1,))
    obtained_prediction = Input(shape=(2,4))
    input = Input(shape=(24,))
    x = Dense(64)(input)
    mid_output = LeakyReLU(0.2)(x)
    x= Dense(20)(mid_output)
    x = LeakyReLU(0.2)(x)
    x1 = Dense(4, activation='tanh')(x)
    x1 = Reshape((1,4))(x1)
    x2 = Dense(4, activation='elu')(x)
    x2 = Lambda(lambda x : -x)(x2)
    x2 = Reshape((1,4))(x2)
    actor = Concatenate(axis=1, name='actor')([x1,x2])
    
    x= Dense(20)(mid_output)
    x = LeakyReLU(0.2)(x)
    critic = Dense(1, name='critic')(x)
    ppo_net = Model(inputs=[input, advantage, obtained_prediction], outputs=[actor, critic]) #the loss_function requires advantage and prediction, so we feed them to the network but keep them unchanged
    ppo_net.compile(optimizer=Adam(1e-3), loss={'actor' : proximal_policy_optimization_loss(advantage,obtained_prediction), 'critic' : 'mean_squared_error'}, 
                    loss_weights={'actor': 0.1, 'critic': 1.})
    if (weights):
        if type(weights)==str:
            ppo_net.load_weights(weights)
        else:
            ppo_net.set_weights(weights)
#    ppo_net.summary()
    return ppo_net

def preprocess_state(state):
    state=np.clip(state, -10, 10)
    return state

def process_reward(reward, state):
    return 0.01*np.clip(reward + (state[14]*np.cos(state[0]) - 0.35), -10,10)

class PPOData:
    states_list: np.array = np.empty((0, 24))
    advantage_list: np.array = np.empty((0, 1))
    predict_list: np.array = np.empty((0, 2, 4))
    action_list: np.array = np.empty((0, 1, 4))
    reward_list: np.array = np.empty((0, 1))
    
    
def train_proc(mem_queue, weight_dict, render):
    #playing process it loads an instance of the model and use it to play the game, then sends the generated batch
    try:
        print('\033[1;34;1mprocess started\033[0m')
        env = gym.make("BipedalWalker-v2")
        import tensorflow as tf
        import tensorflow.keras.backend as K
        #this block enables GPU enabled multiprocessing
        core_config = tf.ConfigProto()
        core_config.gpu_options.allow_growth = True
        tf.logging.set_verbosity(tf.logging.ERROR)
        #allow_growth tells cuda not to use as much VRAM as it wants (as we nneed extra ram for all the other processes)
        with tf.Session(config=core_config) as session:
            K.set_session(session)
            
            #counter of the current version of the weights
            update=0
            #load the initial weights
            ppo_net = create_model(weight_dict['weights'])
            #a generator that plays a game and returns a batch
            while True:
                #stores the weights
                state = preprocess_state(env.reset())
                done=False
                data=PPOData()
                reward_pred=np.empty((0, 1))
                #check for a new update
                if weight_dict['update']>update:
                    #set the counter to the new version
                    update=weight_dict['update']
                    #update the weights
                    ppo_net.set_weights(weight_dict['weights'])
                while not done:
                    data.states_list = np.append(data.states_list, [np.array(state)], axis=0)
                    predict = ppo_net.predict([state.reshape((1,24)), DUMMY_VALUE, DUMMY_ACTION])
                    action=np.clip(np.random.normal(predict[0][0][0], np.exp(predict[0][0][1])), -1, 1)
                    data.action_list = np.append(data.action_list, action.reshape(1,1,4), axis=0)
                    state, reward, done, info = env.step(action)
                    if render:
#                        print(state[14]*np.cos(state[0]))
                        env.render()
                    state=preprocess_state(state)
                    data.predict_list = np.append(data.predict_list, predict[0], axis=0)
                    data.reward_list =np.append(data.reward_list, [np.array([process_reward(reward, state)])], axis=0)
                    reward_pred =np.append(reward_pred, [predict[1][0]], axis=0)
                for i in range(len(data.reward_list)-2, -1, -1):
                    data.reward_list[i]+=data.reward_list[i+1] * GAMMA #compute the discounted obtained reward for each step
                data.advantage_list=data.reward_list-reward_pred
                data.advantage_list -= np.mean(data.advantage_list)
                data.advantage_list /= np.std(data.advantage_list)
                mem_queue.put(([data.states_list, data.advantage_list, data.predict_list], {'actor': data.action_list, 'critic': data.reward_list}))
            K.clear_session()
    except Exception as e: print('\033[1;31;1m', e)


def learn_proc(mem_queue, weight_dict, load, swap_freq=10):
    #learning process it creates or load the model, reads batchs from the player and fit the model with them
    try:
        print('\033[1;34;1mprocess started\033[0m')
        import tensorflow as tf
        import tensorflow.keras.backend as K
        #this block enables GPU enabled multiprocessing
        core_config = tf.ConfigProto()
        #allow_growth tells cuda not to use as much VRAM as it wants (as we nneed extra ram for all the other processes)
        core_config.gpu_options.allow_growth = True
        tf.logging.set_verbosity(tf.logging.ERROR)
        env = gym.make("BipedalWalker-v2")
        with tf.Session(config=core_config) as session:
            K.set_session(session)
            #whether or not to load a previous network
            if(not load):
                ppo_net = create_model()
            else:
                #load the network that scored the best so far
                if os.path.isfile("walker.h5"):
                    ppo_net = create_model("walker.h5")
                else: raise Exception('You need a pretrained net to do this')
            #counter of the current update of the weights
            weight_dict['update']=0
            #stores weights in the global variable for the other processes to access
            weight_dict['weights']=ppo_net.get_weights()
            steps=0
            while True:
                for i in range(swap_freq):
                    steps+=1
                    try:
                        batch, labels = mem_queue.get(False)
                    except Empty:
                        print('\033[1;34;1mlearner playing\033[0m')
                        state = preprocess_state(env.reset())
                        done=False
                        data=PPOData()
                        reward_pred=np.empty((0, 1))
                        while not done:
                            data.states_list = np.append(data.states_list, [np.array(state)], axis=0)
                            predict = ppo_net.predict([state.reshape((1,24)), DUMMY_VALUE, DUMMY_ACTION])
                            action=np.clip(np.random.normal(predict[0][0][0], np.exp(predict[0][0][1])), -1, 1)
                            data.action_list = np.append(data.action_list, action.reshape(1,1,4), axis=0)
                            state, reward, done, info = env.step(action)
                            state=preprocess_state(state)
                            data.predict_list = np.append(data.predict_list, predict[0], axis=0)
                            data.reward_list =np.append(data.reward_list, [np.array([process_reward(reward, state)])], axis=0)
                            reward_pred =np.append(reward_pred, [predict[1][0]], axis=0)
                        for i in range(len(data.reward_list)-2, -1, -1):
                            data.reward_list[i]+=data.reward_list[i+1] * GAMMA #compute the discounted obtained reward for each step
                        data.advantage_list=data.reward_list-reward_pred
                        data.advantage_list -= np.mean(data.advantage_list)
                        data.advantage_list /= np.std(data.advantage_list)
                        batch = [data.states_list, data.advantage_list, data.predict_list]
                        labels = {'actor': data.action_list, 'critic': data.reward_list}
                    print('min std: {}'.format(np.min(batch[2][:,1], axis=0)), 'distance: {}'.format(np.sum(batch[0][:,2], axis=0)), sep='\t') #'max std: ', np.max(batch[2][:,1], axis=0), '\n
                    if steps%1000==0:
                        ppo_net.save_weights("walker_backup_{}.h5".format(steps//1000))
                    ppo_net.fit(batch, labels, batch_size=len(batch[0]), verbose=0)
#                print('\033[1;34;1mupdating net\033[0m')
                weight_dict['weights']=ppo_net.get_weights()
                weight_dict['update']+=1
                #save the weights in a file, to load it later. The file contains the best score ever obtained
                if weight_dict['update']%10==0:
                    print('\033[1;34;1msaving results\033[0m')
                    ppo_net.save_weights("walker.h5")
    except Exception as e: print('\033[1;31;1m', e)
 
def init_worker():
    #Allows to pass a few signals to the processes, such as keyboardInterrupt
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


#import tensorflow as tf
#tf.enable_eager_execution()

def main(load=True, render=1, processes=13): 
    #create shared variables between all the processes
    manager = Manager()
    #contains information about the weights
    weight_dict = manager.dict()
    #a queue of batches to be fed to the training net
    mem_queue = manager.Queue(64)
    
    #initializes all workers
    pool = Pool(processes+1, init_worker)
    try:
        #the learner set the weights and store them in the weight_dict
        pool.apply_async(learn_proc, (mem_queue, weight_dict, load, 1))
        #wait for the learner to finish it's storing
        while 'weights' not in weight_dict:
            time.sleep(0.1)
        for i in range(min(render, processes)):
            pool.apply_async(train_proc, (mem_queue, weight_dict, True))
        for i in range(processes-render):
            #starts the player
            pool.apply_async(train_proc, (mem_queue, weight_dict, False))

        #never currently called, this would be usefull if the processes had an end
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()