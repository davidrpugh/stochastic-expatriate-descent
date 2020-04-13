import typing

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

import experience_replay


class Agent:
    
    def choose_action(self, state: np.array) -> int:
        """Rule for choosing an action given the current state of the environment."""
        raise NotImplementedError
        
    def save(self, filepath) -> None:
        """Save any important agent state to a file."""
        raise NotImplementedError
        
    def step(self,
             state: np.array,
             action: int,
             reward: float,
             next_state: np.array,
             done: bool) -> None:
        """Update agent's state after observing the effect of its action on the environment."""
        raise NotImplmentedError


class DeepQAgent(Agent):

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 number_hidden_units: int,
                 optimizer_fn: typing.Callable[[typing.Iterable[nn.Parameter]], optim.Optimizer],
                 batch_size: int,
                 buffer_size: int,
                 epsilon_decay_schedule: typing.Callable[[int], float],
                 alpha: float,
                 gamma: float,
                 update_frequency: int,
                 double_dqn: bool = False,
                 seed: int = None) -> None:
        """
        Initialize a DeepQAgent.
        
        Parameters:
        -----------
        state_size (int): the size of the state space.
        action_size (int): the size of the action space.
        number_hidden_units (int): number of units in the hidden layers.
        optimizer_fn (callable): function that takes Q-network parameters and returns an optimizer.
        batch_size (int): number of experience tuples in each mini-batch.
        buffer_size (int): maximum number of experience tuples stored in the replay buffer.
        epsilon_decay_schdule (callable): function that takes episode number and returns epsilon.
        alpha (float): rate at which the target q-network parameters are updated.
        gamma (float): Controls how much that agent discounts future rewards (0 < gamma <= 1).
        update_frequency (int): frequency (measured in time steps) with which q-network parameters are updated.
        double_dqn (bool): whether to use vanilla DQN algorithm or use the Double DQN algorithm.
        seed (int): random seed
        
        """
        self._state_size = state_size
        self._action_size = action_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # set seeds for reproducibility
        self._random_state = np.random.RandomState() if seed is None else np.random.RandomState(seed)
        if seed is not None:
            torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # initialize agent hyperparameters
        _replay_buffer_kwargs = {
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "random_state": self._random_state
        }
        self._memory = experience_replay.ExperienceReplayBuffer(**_replay_buffer_kwargs)
        self._epsilon_decay_schedule = epsilon_decay_schedule
        self._alpha = alpha
        self._gamma = gamma
        self._double_dqn = double_dqn
        
        # initialize Q-Networks
        self._update_frequency = update_frequency
        self._online_q_network = self._initialize_q_network(number_hidden_units)
        self._target_q_network = self._initialize_q_network(number_hidden_units)
        self._synchronize_q_networks()
        
        # send the networks to the device
        self._online_q_network.to(self._device)
        self._target_q_network.to(self._device)
        
        # initialize the optimizer
        self._optimizer = optimizer_fn(self._online_q_network.parameters())

        # initialize some counters
        self._number_episodes = 0
        self._number_timesteps = 0
        self._number_parameter_updates = 0
        
    def _initialize_q_network(self, number_hidden_units: int) -> nn.Module:
        """Create a neural network for approximating the action-value function."""
        q_network = nn.Sequential(
            nn.Linear(in_features=self._state_size, out_features=number_hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=number_hidden_units, out_features=number_hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=number_hidden_units, out_features=self._action_size)
        )
        return q_network
    
    def _experiences_to_tensors(self, experiences: typing.List[experience_replay.Experience]):
        tensors = ()
        for i, vs in enumerate(zip(*experiences)):
            if i in {1, 2, 4}:
                tensor = (torch.Tensor(vs)
                               .reshape((-1, 1)))
            if i == 1:
                tensor.long()
            tensors += (tensor.to(self._device),)
        return tensors
    
    def _select_greedy_action(self, states: torch.Tensor, q_network: nn.Module) -> torch.Tensor:
        """Greedy action selection step of Q-learning."""
        _, actions = q_network(states).max(dim=1, keepdim=True)
        return actions
    
    def _evaluate_selected_action(self,
                                  states: torch.Tensor,
                                  actions: torch.Tensor,
                                  rewards: torch.Tensor,
                                  dones: torch.Tensor,
                                  q_network: nn.Module) -> torch.Tensor:
        """Action evaluation step of Q-learning."""
        next_q_values = q_network(states).gather(dim=1, index=actions)        
        q_values = rewards + (self._gamma * next_q_values * (1 - dones))
        return q_values
        
    def _q_learning_update(self,
                           states: torch.Tensor,
                           rewards: torch.Tensor,
                           dones: torch.Tensor,
                           q_network: nn.Module) -> torch.Tensor:
        """Q-learning update rule."""
        actions = self._select_greedy_action(states, q_network)
        q_values = self._evaluate_selected_action(states, actions, rewards, dones, q_network)
        return q_values
    
    def _double_q_learning_update(self,
                                  states: torch.Tensor,
                                  rewards: torch.Tensor,
                                  dones: torch.Tensor,
                                  q_network_1: nn.Module,
                                  q_network_2: nn.Module) -> torch.Tensor:
        """Double Q-learning update rule."""
        actions = self._select_greedy_action(states, q_network_1)
        q_values = self._evaluate_selected_action(states, actions, rewards, dones, q_network_2)
        return q_values
        
    def _learn_from(self, experiences: typing.List[experience_replay.Experience]) -> None:
        """Heart of the DQN with Double Q-Learning algorithm."""
        states, actions, rewards, next_states, dones = (torch.Tensor(vs).to(self._device) for vs in zip(*experiences))
        
        # need to add second dimension to some tensors
        actions = (actions.long()
                          .unsqueeze(dim=1))
        rewards = rewards.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)
        
        if self._double_dqn:
            target_q_values = self._double_q_learning_update(next_states,
                                                             rewards,
                                                             dones,
                                                             self._online_q_network,
                                                             self._target_q_network
                                                            )
        else:
            target_q_values = self._q_learning_update(next_states,
                                                      rewards,
                                                      dones,
                                                      self._target_q_network
                                                      )
            
        # get expected Q values from online model
        online_q_values = (self._online_q_network(states)
                               .gather(dim=1, index=actions))
        # compute the mean squared loss
        loss = F.mse_loss(online_q_values, target_q_values)
        
        # agent updates the parameters of the online network using gradient descent
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        self._soft_update_target_q_network_parameters()
                 
    def _soft_update_target_q_network_parameters(self) -> None:
        """Soft-update of target q-network parameters with the online q-network parameters."""
        for target_param, online_param in zip(self._target_q_network.parameters(), self._online_q_network.parameters()):
            target_param.data.copy_(self._alpha * online_param.data + (1 - self._alpha) * target_param.data)
    
    def _synchronize_q_networks(self) -> None:
        """Synchronize the target_q_network and the local_q_network."""
        _ = self._target_q_network.load_state_dict(self._online_q_network.state_dict())
           
    def _uniform_random_policy(self, state: torch.Tensor) -> int:
        """Choose an action uniformly at random."""
        return self._random_state.randint(self._action_size)
        
    def _greedy_policy(self, state: torch.Tensor) -> int:
        """Choose an action that maximizes the action_values given the current state."""
        # evaluate the network to compute the action values
        self._online_q_network.eval()
        with torch.no_grad():
            action_values = self._online_q_network(state)
        self._online_q_network.train()
        
        # choose the greedy action
        action = (action_values.cpu()  # action_values might reside on the GPU!
                               .argmax()
                               .item())
        return action
    
    def _epsilon_greedy_policy(self, state: torch.Tensor, epsilon: float) -> int:
        """With probability epsilon explore randomly; otherwise exploit knowledge optimally."""
        if self._random_state.random() < epsilon:
            action = self._uniform_random_policy(state)
        else:
            action = self._greedy_policy(state)
        return action

    def choose_action(self, state: np.array) -> int:
        """
        Return the action for given state as per current policy.
        
        Parameters:
        -----------
        state (np.array): current state of the environment.
        
        Return:
        --------
        action (int): an integer representing the chosen action.

        """
        # need to reshape state array and convert to tensor
        state_tensor = (torch.from_numpy(state)
                             .unsqueeze(dim=0)
                             .to(self._device))
            
        # choose uniform at random if agent has insufficient experience
        if not self.has_sufficient_experience():
            action = self._uniform_random_policy(state_tensor)
        else:
            epsilon = self._epsilon_decay_schedule(self._number_episodes)
            action = self._epsilon_greedy_policy(state_tensor, epsilon)
        return action
    
    def has_sufficient_experience(self) -> bool:
        """True if agent has enough experience to train on a batch of samples; False otherwise."""
        return len(self._memory) >= self._memory.batch_size
    
    def save(self, filepath: str) -> None:
        """
        Saves the state of the DeepQAgent.
        
        Parameters:
        -----------
        filepath (str): filepath where the serialized state should be saved.
        
        Notes:
        ------
        The method uses `torch.save` to serialize the state of the q-network, 
        the optimizer, as well as the dictionary of agent hyperparameters.
        
        """
        checkpoint = {
            "q-network-state": self._online_q_network.state_dict(),
            "optimizer-state": self._optimizer.state_dict(),
            "agent-hyperparameters": {
                "alpha": self._alpha,
                "batch_size": self._memory.batch_size,
                "buffer_size": self._memory.buffer_size,
                "gamma": self._gamma,
                "update_frequency": self._update_frequency
            }
        }
        torch.save(checkpoint, filepath)
        
    def step(self, state: np.array, action: int, reward: float, next_state: np.array, done: bool) -> None:
        """
        Updates the agent's state based on feedback received from the environment.
        
        Parameters:
        -----------
        state (np.array): the previous state of the environment.
        action (int): the action taken by the agent in the previous state.
        reward (float): the reward received from the environment.
        next_state (np.array): the resulting state of the environment following the action.
        done (bool): True is the training episode is finised; false otherwise.
        
        """
        # save experience in the experience replay buffer
        experience = experience_replay.Experience(state, action, reward, next_state, done)
        self._memory.append(experience)
            
        if done:
            self._number_episodes += 1
        else:
            self._number_timesteps += 1
            
            # every so often the agent should learn from experiences
            if self._number_timesteps % self._update_frequency == 0 and self.has_sufficient_experience():
                experiences = self._memory.sample()
                self._learn_from(experiences)
