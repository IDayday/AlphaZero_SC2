# AlphaZeroSC2

>The first time, I want to try this by using the pySC2, but it is hard to create a zero-sum game in mini-game buildmarines. So I have to restructure the whole game by myself, you can see the details in bmgame.py.

>The mini-game buildmarines in order to solve an optimization problem of the production, but it only has one player. If I set another player in the game, and set some limit of the parameters, it can be transferred to a zero-sum game environment. And I can try to use MCTS to solve this problem.

## 5.21

🙄🙄🙄

**Say Goodbye to the pySC2** (PS: it wastes too much of my time, nearly a month)

😚😚😚

**Say Hello to my bmgame** (PS: I try my best to restructrue the whole game logic, also about the inner parameters)

Maybe I need to explain why I want to do this. First, I like the SC2 game😋. Second, it's my course work 😥(solve a zero-sum game by using MCTS).

## 5.22

It's easy for me to restart at any obs/node as I want. I set the env(bmgame) to be out of the agent unlike it was before. The obs which the env return combines everything about the current state of the env. So you can easily copy obs if you want to store it in the child node.

* The dqn_model and dqn_agent are completed!
* The MCTS is over a half.

## 5.23

😘😘😘

I have finished the obvious bugs, and now, training can be started.
I test some epoch, the simulation results most likely the draw. The agent doesn't know how to build marines in current policy. 

* For the setting default, simulation of a whole game cost about 3mins.

Maybe, parallel training set is difficult for me now. I will try if I have a time. I need to finish the report first.

The training log just like below, the **simulation result[-2]** means the reward. So "[0,0]" means draw. The total steps of the game are 500. Every search in the one step of simulation about 200 times. The time cost of a whole game can be seen in line **simulation i cost xxx second**.

~~~bash
simulation start : 2/500
simulation result :  [[2650.0, 1.25, 4350.0], [135, 12, 12, 0], [10, 1], [[], [], [], []], [2800.0, 1.25, 4350.0], [135, 12, 12, 0], [10, 0], [[], [], [], []], [0, 0], [250, 250]]
simulation 2 cost 116.15 second
simulation is over!
start training!
epoch 0 batchloss:  0.013598231598734856
epoch 1 batchloss:  0.0140206478536129
epoch 2 batchloss:  0.015985790640115738
epoch 3 batchloss:  0.01822245679795742
epoch 4 batchloss:  0.011359521187841892
simulation start : 3/500
simulation result :  [[4182.0, 1.2, 5982.0], [123, 19, 19, 0], [9, 0], [[], [], [], []], [3572.0, 1.25, 5422.0], [147, 20, 16, 0], [11, 0], [[], [], [], []], [0, 0], [250, 250]]
simulation 3 cost 107.9 second
simulation is over!
start training!
epoch 0 batchloss:  0.008436528965830803
epoch 1 batchloss:  0.007282668724656105
epoch 2 batchloss:  0.007358501199632883
epoch 3 batchloss:  0.007472369819879532
epoch 4 batchloss:  0.006356597878038883
~~~

## 5.24

There is a trouble in the training now! The agent seems to fall into local optimality. As shown in the log, at first, it doesn't know how to build the depot, so the population of SCV is stuck at 15. Then, it start to build the depot for increasing total population, in order to build more SCV to accelerate the collecting of mainral. But it is far from the goal of building marines.

### test
* The reward is no longer divided by the trajectory step.
* Increase the warm-up phase.
* Change the Loss design, it means to use other algorithem no longer DQN.
* Change the default game parameters: 
    * total steps 500 -> 600
    * mineral rate function
* VS random opponent agent
* Add noise to the action prob from the model predict result
* Add intrinsic reward                                          

## 5.25

I made a big mistake, when the agent generated data, it needed a randomicity, for example, random.choice from action prob , or add a noise to th action prob and sample the maxprob. If don't do this, the model will be stable and the choice in each step will be fixed. So you will see the same result of the game simulation.

Another question is that I'm not clear how to keep the policy optimized monotonously.

**A simple parallelization method I implemented is through the Pipe function. It doesn't accelerate the rate of simulation but can get more trans data from parallelized simulation in one time.** 

## 5.26

I have encountered something strange, my multi-threaded simulation results are exactly the same, whether I set a global random seed or an in-thread random seed, the results of both threads are always the same, and the results of multiple iterations are also the same, I am currently still not figuring out what the reason is, I plan to rewrite the parallelization code.

~~~bash
Running on cpu and 2 processes
simulation start : 1/500
parent process: 790
process id: 947
new obs getting
get trans from 0 process
parent process: 790
process id: 948
new obs getting
simulation result : [[32.0, 30.0, 7732.0], [159, 104, 63, 27], [12, 7], [[6, 11, 14], [2, 3, 4, 6, 7, 10, 16, 20, 23, 24, 25], [], [32, 47]], [87.5, 30.0, 7287.5], [111, 111, 65, 37], [8, 5], [[2, 5, 7], [3, 6, 11, 14, 17, 22], [], [59]], [27, 37], [289, 288]] 

get trans from 1 process
simulation result : [[32.0, 30.0, 7732.0], [159, 104, 63, 27], [12, 7], [[6, 11, 14], [2, 3, 4, 6, 7, 10, 16, 20, 23, 24, 25], [], [32, 47]], [87.5, 30.0, 7287.5], [111, 111, 65, 37], [8, 5], [[2, 5, 7], [3, 6, 11, 14, 17, 22], [], [59]], [27, 37], [289, 288]] 

stop process
simulation 1 cost 105.25 second
simulation is over!
simulation start : 2/500
parent process: 790
process id: 1121
new obs getting
get trans from 0 process
parent process: 790
process id: 1122
new obs getting
simulation result : [[32.0, 30.0, 7732.0], [159, 104, 63, 27], [12, 7], [[6, 11, 14], [2, 3, 4, 6, 7, 10, 16, 20, 23, 24, 25], [], [32, 47]], [87.5, 30.0, 7287.5], [111, 111, 65, 37], [8, 5], [[2, 5, 7], [3, 6, 11, 14, 17, 22], [], [59]], [27, 37], [289, 288]] 

get trans from 1 process
simulation result : [[32.0, 30.0, 7732.0], [159, 104, 63, 27], [12, 7], [[6, 11, 14], [2, 3, 4, 6, 7, 10, 16, 20, 23, 24, 25], [], [32, 47]], [87.5, 30.0, 7287.5], [111, 111, 65, 37], [8, 5], [[2, 5, 7], [3, 6, 11, 14, 17, 22], [], [59]], [27, 37], [289, 288]] 

stop process
simulation 2 cost 105.11 second
simulation is over!
~~~

## 5.28

🏆🏆🏆
**Congradulation myself, I have finished the multiprocesses evaluating program.**

* The "top" reflect the CPU used information as follow:
<img src='top.jpg' width=600>

* evaluating 45 checkpoints needs 1735 seconds
* The evaluting results as follow:

<img src='eva_result.jpg' width=600>

Where the number in the axis means the iterations of checkpoint. And the number in the box means win rate. The performance getting better.

👊👊👊  
**Tonight belongs to UEFA Champions League**  
**Tomorrow belongs to LOL MSI**

## TODO
* ~~reset MCTS in new game environment~~
* ~~train set in new game environment~~
* ~~maybe the parallel training set~~