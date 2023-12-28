# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Value Iteration Formula: V(s) = max_a Q(s, a), we want to find the argmax_a action
        states = self.mdp.getStates()
        for _ in range(self.iterations):
            updated_values = util.Counter()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                if len(actions) == 0 or self.mdp.isTerminal(state): continue
                updated_values[state] = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
            self.values = updated_values
            
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Q(s, a) = sum_s' T(s, a, s')[R(s, a, s') + gamma * V(s')], gamma is the discount factor self.discount
        transition_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        QValue = 0.0
        for transition, prob in transition_probs:
            QValue += prob * (self.mdp.getReward(state, action, transition) + self.discount * self.getValue(transition))

        return QValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Value Iteration Formula: V(s) = max_a Q(s, a), we want to find the argmax_a action
        if self.mdp.isTerminal(state): return None
        actions = self.mdp.getPossibleActions(state)
        # print(actions)
        qs = util.Counter()
        for action in actions:
            qs[action] = self.computeQValueFromValues(state, action)
        return qs.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for _ in range(self.iterations):
            state = states[_ % len(states)]
            actions = self.mdp.getPossibleActions(state)
            if len(actions) == 0 or self.mdp.isTerminal(state): continue
            self.values[state] = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
            

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Compute predecessors of all states
        predecessors = {}
        states = self.mdp.getStates()
        for state in states:
            predecessors[state] = set()
        for state in states:
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                transition_probs = self.mdp.getTransitionStatesAndProbs(state, action)
                for transition, prob in transition_probs:
                    if prob > 0:
                        predecessors[transition].add(state)
        # Initialize an empty priority queue
        min_heap = util.PriorityQueue()
        # For each non-terminal state s, do:
        # autograder works only when iterating over states in order
        for state in states:
            if self.mdp.isTerminal(state): continue
            # Find the absolute value of the diff between current value of s in self.values 
            # and the highest Q-value across all possible actions from s, call this number diff
            highest_q = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
            diff = abs(self.values[state] - highest_q)
            # Push s into the priority queue with priority -diff, because the pq is a min heap, so if we want to 
            # make highest diff first, we need to push -diff into the pq
            min_heap.push(state, -diff)
        # For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for _ in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if min_heap.isEmpty(): return
            # Pop a state s off the priority queue.
            state = min_heap.pop()
            # Update s's value (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(state):
                self.values[state] = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
            # For each predecessor p of s, do:
            for pre in predecessors[state]:
                # Find the absolute value of the diff between current value of p in self.values 
                # and the highest Q-value across all possible actions from p, call this number diff
                highest_q = max([self.getQValue(pre, action) for action in self.mdp.getPossibleActions(pre)])
                diff = abs(self.values[pre] - highest_q)
                # If diff > theta, push p into the priority queue with priority -diff
                if diff > self.theta: min_heap.update(pre, -diff)

