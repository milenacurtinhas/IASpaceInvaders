import numpy as np
from abc import ABC, abstractmethod
from typing import List

class Agent(ABC):
    """Interface para todos os agentes."""
    @abstractmethod
    def predict(self, state: np.ndarray) -> int:
        """Faz uma previsão de ação com base no estado atual."""
        pass

class HumanAgent(Agent):
    """Agente controlado por um humano (para modo manual)"""
    def predict(self, state: np.ndarray) -> int:
        # O estado é ignorado - entrada vem do teclado
        return 0  # Padrão: não fazer nada (será sobrescrito pela entrada do usuário no manual_play.py)

# Funções de ativação para a rede neural
def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class SimpleNeuralNetwork:
    """Rede neural totalmente conectada com bias"""
    def __init__(self, input_size=27, hidden_sizes=[32, 16], output_size=3):
        # Inicialização dos pesos (com bias)
        # Camada 1: entrada (27) + bias (1) -> 32 neurônios
        # Camada 2: 32 + bias (1) -> 16 neurônios  
        # Camada 3: 16 + bias (1) -> 3 neurônios de saída
        self.layer_sizes = [
            (input_size + 1, hidden_sizes[0]),    # 28 x 32
            (hidden_sizes[0] + 1, hidden_sizes[1]), # 33 x 16
            (hidden_sizes[1] + 1, output_size)     # 17 x 3
        ]
        
        # Inicialização aleatória dos pesos
        self.layers = []
        for size in self.layer_sizes:
            self.layers.append(np.random.randn(size[0], size[1]) * 0.1)

    def forward(self, x):
        """Propagação direta"""
        # Primeira camada oculta
        x = np.append(x, 1.0)  # adiciona bias
        z1 = tanh(np.dot(x, self.layers[0]))
        
        # Segunda camada oculta
        z1 = np.append(z1, 1.0)  # adiciona bias
        z2 = tanh(np.dot(z1, self.layers[1]))
        
        # Camada de saída
        z2 = np.append(z2, 1.0)  # adiciona bias
        output = softmax(np.dot(z2, self.layers[2]))
        
        return output

    def get_weights(self):
        """Retorna todos os pesos como um vetor unidimensional"""
        return np.concatenate([w.flatten() for w in self.layers])

    def set_weights(self, weights):
        """Atualiza a rede a partir de um vetor de pesos"""
        idx = 0
        self.layers = []
        
        for size in self.layer_sizes:
            w_size = np.prod(size)
            self.layers.append(weights[idx:idx+w_size].reshape(size))
            idx += w_size

class NeuralNetworkAgent(Agent):
    """Agente baseado em rede neural"""
    def __init__(self, weights=None):
        self.model = SimpleNeuralNetwork()
        if weights is not None:
            self.model.set_weights(weights)
    
    def predict(self, state: np.ndarray) -> int:
        """Prediz a ação baseada no estado atual"""
        output = self.model.forward(state)
        return int(np.argmax(output))

class BatAlgorithmTrainer:
    """Implementação do Algoritmo dos Morcegos para otimização"""
    def __init__(self, pop_size, num_iterations, game_eval_fn,
                 weight_dim, fmin=0, fmax=2, alpha=0.9, gamma=0.9):
        self.pop_size = pop_size
        self.num_iterations = num_iterations
        self.game_eval_fn = game_eval_fn
        self.weight_dim = weight_dim
        self.fmin = fmin
        self.fmax = fmax
        self.alpha = alpha
        self.gamma = gamma

    def optimize(self):
        """Executa uma iteração do algoritmo dos morcegos"""
        # Inicialização da população
        bats = np.random.uniform(-1, 1, (self.pop_size, self.weight_dim))
        velocities = np.zeros_like(bats)
        freq = np.zeros(self.pop_size)
        loudness = np.random.uniform(1, 2, self.pop_size)  # A0
        pulse_rate = np.random.uniform(0, 1, self.pop_size)  # r0

        # Avaliação inicial
        print("Avaliando população inicial...")
        fitness = np.array([self.game_eval_fn(b) for b in bats])
        best_idx = np.argmax(fitness)
        best_bat = bats[best_idx].copy()
        best_fit = fitness[best_idx]

        print(f"Melhor fitness inicial: {best_fit:.2f}")

        for t in range(self.num_iterations):
            for i in range(self.pop_size):
                # Atualiza frequência (Eq. 2 do artigo)
                beta = np.random.rand()
                freq[i] = self.fmin + (self.fmax - self.fmin) * beta
                
                # Atualiza velocidade e posição (Eq. 3 e 4)
                velocities[i] = velocities[i] + (bats[i] - best_bat) * freq[i]
                new_bat = bats[i] + velocities[i]

                # Busca local (Eq. 5)
                if np.random.rand() > pulse_rate[i]:
                    # Gera solução local ao redor da melhor atual
                    epsilon = np.random.uniform(-1, 1, self.weight_dim)
                    avg_loudness = np.mean(loudness)
                    new_bat = best_bat + epsilon * avg_loudness

                # Avaliação da nova solução
                new_fitness = self.game_eval_fn(new_bat)

                # Critério de aceitação
                if np.random.rand() < loudness[i] and new_fitness > fitness[i]:
                    bats[i] = new_bat
                    fitness[i] = new_fitness
                    
                    # Atualiza loudness e pulse rate (Eq. 6)
                    loudness[i] *= self.alpha
                    pulse_rate[i] = pulse_rate[i] * (1 - np.exp(-self.gamma * t))

                # Atualiza melhor global
                if new_fitness > best_fit:
                    best_fit = new_fitness
                    best_bat = new_bat.copy()

        return best_bat, best_fit