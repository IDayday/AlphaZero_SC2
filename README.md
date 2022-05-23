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
## TODO
* ~~reset MCTS in new game environment~~
* ~~train set in new game environment~~
* maybe the parallel training set