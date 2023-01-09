# MBPO-MDP

## What does this repo contains :

- We have built a framework that consist of an MBPO(A model free RL algorithm) based Agent interacting with an environment,whose dynamics are given by a Tabular MDP.This can strengthen our undertanding on what exactly is going on with the MBPO algorithm. Since optimal policy can be explicitly computed for tabular MDP's ,we can do a comparison with what MBPO based agent gives.In short,the user can use this for debugging purposes.

## Objective : <br/>
- To learn/explore about the Real data vs Fake data trade off,that exist in Model Based Policy Optimisation (MBPO) algorithm. 

## Pre-requisites:
Basic understanding the following topics are expected:

- MBPO algorithm.
- off policy policy gradient(importance sampling based).
- MDPs 

I will touch upon MBPO algorithm,a practical version of the same and the intuition behind MBPO algorithm.

## Algorithm proposed :(Simplified version)
<img width="583" alt="Screenshot 2022-11-17 at 12 07 22 AM" src="https://user-images.githubusercontent.com/113635391/202267312-78099037-df2e-4f5a-8b32-37feb8cb9192.png">



## Algorithm proposed:(Implementation version)

<img width="579" alt="Screenshot 2022-11-17 at 12 07 33 AM" src="https://user-images.githubusercontent.com/113635391/202267366-fd1939e5-68d9-4440-a84b-e7cd288af667.png">

## Intuition :

MBPO optimizes a policy under a learned model, collects data under the updated policy, and uses that data to train a new
model.



## Agent can be switched to any of the following training approaches:

### A Note before:

- By Real episodes ,I mean the episodes obtained by interacting with the actual environment.
- By Fake episodes ,I mean the the episode obtained by interacting with the estimated environment.As per MBPO algorithm,the agent will maintain an estimate of the dynamics of the environment,now using this estimate ,agent will generate the a bunch of fake trajectories.then perform policy gradient on this fake data.

### Three approaches:

- Policy gradient on real episodes(pure):
        This is same as that of the normal off -policy policy gradient.
- Policy gradient on fake episodes(pure):
        Here policy gradient will be performed but using the fake episode data.
- Policy gradient on a mixture of real and fake episodes(mixed):
    Here policy gradient will be performed using data buffer with comprise of a mixture of real and fake episodes.The user can specify the mixing ratio as a parameter.


### Thanks/References:
- https://github.com/BlackHC/mdp : A python library for implementing MDPs that go along with the openAI gym framework.