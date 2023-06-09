from datetime import datetime
from os import mkdir

import gym
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

from gym import envs
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel, RationalQuadratic, ExpSineSquared
import argparse
dirName= None


def create_train_test(samples):
    '''
    Create train and test sets for
    :param s:
    :return:
    '''
    X = np.zeros((len(samples),3))
    Y = np.zeros((len(samples),2))

    for i in range(len(samples)):
        s, a, s_p = samples[i]
        x_i = np.array([s[0],s[1],a])
        y_i = np.array([s_p[0],s_p[1]])
        X[i] = x_i
        Y[i] = y_i

    return X,Y


class GPRL:
    '''
    This implementation of GPRL is assuming that the algorthim
    has been passed an enivironment obeject of the form used in the
    OPEN AI GYM implementations. I will be using the MountianCar Environment
    '''

    def __init__(self,env,gamma, draw= False):
        '''
        :param env: Gym style environment
        '''
        self.actions = [0,1,2]
        self.env = env
        self.gamma = gamma # future value discount rate

        self.GP_V = None # Value GP

        self.draw= draw


        self.pos_m = np.array([]) # Pos values
        self.vel_m = np.array([]) # velocity values
        self.S = np.array([])
        self.V = np.array([]) # Values lookup Table

    def act_greedy(self, s):
        '''
        From current state
        get max action
        :param s:
        :return:
        '''

        self.env.reset()
        a_max = None
        a_best_v = -999
        for a in self.actions:

            self.env.env.state = s

            s_p,r,d,n = self.env.step(a)

            if r < 0: r = 0

            if d == True:
                r = 1

            s_p = s_p.reshape((1,s_p.shape[0]))
            v_s = self.GP_V.predict(s_p)[0][0]

            a_v = r + self.gamma * v_s

            if a_v > a_best_v:
                a_max = a
                a_best_v = a_v

        return a_max


    def create_grid(self,n=25):
        '''
        Create 2d grid of position,velocity values
        '''
        min_pos = self.env.min_position
        max_pos = self.env.max_position
        max_speed = self.env.max_speed

        self.pos_m = np.linspace(min_pos,max_pos,n)
        self.vel_m = np.linspace(-max_speed,max_speed,n)
        V = np.zeros((n,n))
        S = np.zeros((n,n,2))

        for i, x in enumerate(self.pos_m):
            for j, dx in enumerate(self.vel_m):
                S[i,j] = np.array([x,dx])
                V[i,j] = self.sample_env(x,dx)

        return S,V

    def init_value(self,m=21):
        '''
        Sample m support vectors and initialize
        V (m x d) = (V(s_1) ... V(s_m)) where V(s_i) = R_i
        Where R_i is given by the environment or sampled
        from the system dynamics and then computed using equation (7) from paper
        :param m: Number of support vectors
        :return:
        '''

        # For now just creating a grid of values initilized to
        # the reward at that point in the environment
        S,V = self.create_grid(m)
        self.S = S
        self.V = V



    def plot_best_path(self, path):
        '''
        Function for taking the current value function
        and plotting a graph of the path taken through
        space.
        :return:
        '''


        y = [x for x in range(1,len(path) + 1)]

        plt.plot(path,y)
        plt.scatter(path,y)
        plt.xlim(self.env.min_position,self.env.max_position)

        plt.title('Best path')
        #plt.show()

        plt.savefig('{}/best_path.png'.format(dirName))
        plt.close()


    def plot_value_func(self,V,text=''):
        '''
        Plot the Value matrix in 3D
        :param V: Matrix were rows are position and columns are velocity
        :return:
        '''

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        pos, vel = np.meshgrid(self.pos_m, self.vel_m)
        surf = ax.plot_surface(pos, vel, V.T,cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(0, 5)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.title(text)
        #plt.show()
        plt.savefig('{}/value_func.png'.format(dirName))
        plt.close(fig)

    def run(self,T=50):
        '''
        Steps of algorithm 1 from paper
            1-2 Modeling System dynamics
            3: Init Value function by sampling m support points
            4: Policy Iteration
        :param T: Number of steps to run per episode
        :return:
        '''

        ######################
        # Compute GP for ENV
        ######################


        self.init_value(m=21)

        self.env.reset()

        N = self.V.shape[0]

        self.W = np.zeros((N**2,N**2))

        Y = self.V.reshape((N**2,1)) #init V with R

        S = self.S.reshape((N**2,2))

        GRID_SIZE = 100
        S_Grid,_ = self.create_grid(GRID_SIZE)

        self.GP_V = self.GP_V.fit(S, Y)

        S_Grid = S_Grid.reshape((GRID_SIZE**2,2))



        for t in range(T):
            R = np.zeros((S.shape[0],1))
            V = np.zeros((S.shape[0],1))
            for i, s_i in enumerate(S[::-1]):

                a = self.act_greedy(s_i)

                self.env.env.state = s_i

                s,r,d,_ = self.env.step(a)

                if r < 0:
                    r = 0

                if d:
                    r = 1

                R[i] = r

                s = s.reshape((1,s.shape[0]))

                v_s = self.GP_V.predict(s)[0][0]

                V[i] = r + self.gamma*v_s


            self.V = V.reshape((N,N))

            self.GP_V = self.GP_V.fit(S, V)



            path, success, statep = self.test_env()
            #if success:
                #print(t, statep)
                #break

            #self.plot_value_func(V_s, 'Value at iteration {}'.format(t))
        if self.draw:

            path= self.simulate_env()
            self.plot_best_path(path)
            y_pred = self.GP_V.predict(S_Grid)
            V_s = y_pred.reshape((GRID_SIZE, GRID_SIZE))
            self.plot_value_func(V_s,'Value at iteration {}'.format('Final'))

        all_path=[]

        for i in range(50):
            path, success, statep = self.test_env()
            all_path.append(len(path))
            #print(len(path))
        #print(np.mean(all_path))
        #print(np.std(all_path))
    def sample_discreet_env(self,M):
        '''
        Function to randomly grab samples from
        the environment
        :param env: Gym object
        :return:
        '''

        min_pos = self.env.min_position
        max_pos = self.env.max_position
        max_speed = self.env.max_speed
        goal_position = self.env.goal_position
        goal_velocity = self.env.goal_velocity

        samples = []

        for n in range(M):
            sample_pos = np.random.uniform(min_pos, max_pos)

            sample_vel = np.random.uniform(0, max_speed)

            s = (sample_pos, sample_vel)

            action = np.random.randint(3)

            self.env.env.state = s  # set gym environment to the state to sample from

            r1 = bool(sample_pos >= goal_position and sample_vel >= goal_velocity)

            self.env.step(action)

            s_p = self.env.env.state  # new state from the environment

            r2 = bool(s_p[0] >= goal_position and s_p[1] >= goal_velocity)

            samples.append((s, r1, action, s_p,r2))

        return samples

    def sample_env(self, p, v):
        '''
        :param p:
        :param v:
        :return:
        '''

        r1 = bool(p >= self.env.goal_position and v >= self.env.goal_velocity)

        return r1

    def test_env(self):

        env= self.env
        env.reset()
        #env = env.unwrapped
        #rndPos= -np.random.uniform(0.52, 0.58)
        #env.state= np.array([rndPos, 0])
        state= env.state
        pos_x = []
        success= False
        for _ in range(200):

            action_r = self.act_greedy(env.state)

            s,r,d,_ = env.step(action_r)  # take a random action
            #print(s, action_r)
            pos_x.append(s[0])
            if d:
                success= True
                break

        self.env.close()

        return pos_x, success, state

    def simulate_env(self):
        '''
        Run the actual gym enviroment
        to visualize how it performs
        :param N:
        :return:
        '''
        env_wrap = gym.wrappers.Monitor(self.env, dirName, force=True)

        env_wrap.reset()

        #rndPos = -np.random.uniform(0.52, 0.58)
        #env_wrap.env.env.state = np.array([-0.57, 0])
        pos_x = []

        for _ in range(200):


            action_r = self.act_greedy(env_wrap.state)

            s,r,d,_ = env_wrap.step(action_r)  # take a random action

            pos_x.append(s[0])
            if d:
                break


        self.env.close()
        env_wrap.close()

        return pos_x


if __name__ == '__main__':


    # create the parser
    parser = argparse.ArgumentParser()

    # add arguments to the parser
    parser.add_argument('-T', '--T', type=int, help="Number of runs", default= 10)
    parser.add_argument('-ker', '--kernel', type=str, help="Kernel", default= 'Matern')
    parser.add_argument('-g', '--gamma', type=float, help="Discount", default= 0.8)
    parser.add_argument('-p', '--plot', type=bool, help="Draw the plots or not", default= False)
    parser.add_argument('-dir', '--dir', type=str, help="directory to save", default= "Data")

    # parse the arguments
    args = parser.parse_args()

    # Hyperparameters
    T = args.T
    gamma = args.gamma
    draw = args.plot

    dirName= args.dir



    if args.kernel == 'Matern':
        kernel = Matern(length_scale=2, nu=3 / 2)+ ConstantKernel(constant_value=2) + WhiteKernel(noise_level=1)
    elif args.kernel == 'RBF':
        kernel = RBF(1) + ConstantKernel(constant_value=2)+ WhiteKernel(noise_level= 1)
    elif args.kernel == 'RationalQuadratic':
        kernel = RationalQuadratic(length_scale=1.0, alpha=1.5) + ConstantKernel(constant_value=2) + WhiteKernel(noise_level=1)
    elif args.kernel == 'ExpSineSquared':
        kernel = ExpSineSquared(length_scale=1.0, periodicity=4, periodicity_bounds=(1e-2, 1e1)) + ConstantKernel(constant_value=2) + WhiteKernel(noise_level=1)
    else:
        kernel = Matern(length_scale=2, nu=3 / 2)+ ConstantKernel(constant_value=2) + WhiteKernel(noise_level=1)




    env = gym.make('MountainCar-v0')
    env.reset()

    gprl = GPRL(env,gamma= gamma, draw= draw)

    gp = GaussianProcessRegressor(kernel=kernel, optimizer= "fmin_l_bfgs_b", n_restarts_optimizer=0, random_state= 2002)

    gprl.GP_V = gp

    gprl.run(T=T)

    min_pos, max_pos = -1.2, 0.6
    min_vel, max_vel = -0.07, 0.07

    # Discretize state space
    sample_pos = np.linspace(min_pos, max_pos, 5)
    sample_v = np.linspace(min_vel, max_vel, 5)

    for pos in sample_pos:
        for v in sample_v:
            state = np.array([pos, v])
            _, std= gp.predict(state.reshape(1, -1), return_std=True)
            print(std[0])

    print("///////////////////////////////")

