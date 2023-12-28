# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.Q = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.Q[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # V(s) = max_a Q(s,a)
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) == 0:
            return 0.0
        else:
            return max([self.getQValue(state, action) for action in legal_actions])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legal_actions = self.getLegalActions(state)
        if len(legal_actions) == 0:
            return None
        else:
            # argmax_a Q(s,a)
            return max(legal_actions, key=lambda action: self.getQValue(state, action))
    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if len(legal_actions) == 0:
            return None
        return random.choice(legal_actions) if util.flipCoin(self.epsilon) else self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.Q[(state, action)] = (1 - self.alpha) * self.Q[(state, action)] + self.alpha * sample
        

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self,
                 epsilon=0.05,
                 gamma=0.8,
                 alpha=0.2,
                 numTraining=0,
                 **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        feature_vector = self.featExtractor.getFeatures(state, action)
        return sum([self.weights[feature] * feature_vector[feature] for feature in feature_vector])

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        feature_vector = self.featExtractor.getFeatures(state, action)
        # difference = (r + gamma * max_a' Q(s',a')) - Q(s,a) = (r + gamma * V(s')) - Q(s,a)
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        
        for feature in feature_vector:
            # w_i = w_i + alpha * difference * f_i(s,a)
            # self.weights[feature] = self.weights[feature] + self.alpha * difference * feature_vector[feature]
            self.weights[feature] += self.alpha * difference * feature_vector[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass


class BetterExtractor(FeatureExtractor):
    "Your extractor entry goes here.  Add features for capsuleClassic."

    def getFeatures(self, state, action):
        features = SimpleExtractor().getFeatures(state, action)
        # Add more features here
        "*** YOUR CODE HERE ***"
        features['bias_term'] = 1.0
        features['num_of_foods'] = len(state.getFood().asList())
        features['num_of_capsules'] = len(state.getCapsules())
        features['num_of_ghosts'] = len(state.getGhostPositions())
        features['closest_food'] = min([util.manhattanDistance(state.getPacmanPosition(), food) for food in state.getFood().asList()]) if len(state.getFood().asList()) > 0 else 0
        features['closest_capsule'] = min([util.manhattanDistance(state.getPacmanPosition(), capsule) for capsule in state.getCapsules()]) if len(state.getCapsules()) > 0 else 0
        features['closest_ghost'] = min([util.manhattanDistance(state.getPacmanPosition(), ghost) for ghost in state.getGhostPositions()]) if len(state.getGhostPositions()) > 0 else 0
        features['closest_food_reciprocal'] = 1 / features['closest_food'] if features['closest_food'] > 0 else 0
        features['closest_capsule_reciprocal'] = 1 / features['closest_capsule'] if features['closest_capsule'] > 0 else 0
        features['closest_ghost_reciprocal'] = 1 / features['closest_ghost'] if features['closest_ghost'] > 0 else 0
        features['score'] = state.getScore()
        scared_ghosts = filter(lambda g: g.scaredTimer > 0, state.getGhostStates())
        return features
