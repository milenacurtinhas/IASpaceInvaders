import numpy as np
from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent
from bat_algorithm import BatAlgorithm
from test_trained_agent_bat import test_agent
import os
import sys
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

POPULATION_SIZE = 100
GENERATIONS = 1000
MAX_TIME = 12 * 3600  # 12 horas

def game_fitness_function(population: np.ndarray) -> np.ndarray:
    """Função de fitness que avalia múltiplos indivíduos"""
    game_config = GameConfig(num_players=len(population), fps=60)
    agents = [NeuralNetworkAgent(weights) for weights in population]
    total_scores = np.zeros(len(agents))
    
    # Executa 3 jogos para cada indivíduo
    for i in range(3):
        game = SurvivalGame(config=game_config, render=False)
        
        while not game.all_players_dead():
            actions = []
            for idx, agent in enumerate(agents):
                if game.players[idx].alive:
                    state = game.get_state(idx, include_internals=True)
                    action = agent.predict(state)
                    actions.append(action)
                else:
                    actions.append(0)
            game.update(actions)
            
            if game.render:
                game.render_frame()
        
        for idx, player in enumerate(game.players):
            total_scores[idx] += player.score
    
    average_scores = total_scores / 3    
    return average_scores

def train_and_test():
    print("\n--- Iniciando Treinamento com Bat Algorithm ---")
    
    # Dimensão dos pesos da rede neural
    weight_dim = 28*32 + 33*16 + 17*3  # 1475 pesos
    
    ba = BatAlgorithm(
        population_size=POPULATION_SIZE,
        weight_dim=weight_dim,
        fmin=0,
        fmax=2,
        alpha=0.9,
        gamma=0.9
    )
    
    best_weights_overall = None
    best_fitness_overall = -np.inf
    evolution_scores = []
    
    start_time = time.time()
    
    for generation in range(GENERATIONS):
        # Verifica limite de tempo
        if (time.time() - start_time) > MAX_TIME:
            print("Interrompido por tempo (12h).")
            break
        
        start_generation = time.time()
        current_best_weights, current_best_fitness = ba.evolve(game_fitness_function, parallel=True)
        
        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_weights_overall = current_best_weights
            print(f'Backup generation -> Melhor Fitness Geral: {best_fitness_overall:.2f}')
            np.save("best_bat_weights.npy", best_weights_overall)
        
        evolution_scores.append(best_fitness_overall)
        end = time.time()
        
        print(f"{generation + 1}/{GENERATIONS} Best Fitness: {current_best_fitness:.2f} Melhor Fitness Geral: {best_fitness_overall:.2f} ({end-start_generation:.2f} s)")
    
    print("\n--- Treinamento Concluído ---")
    print(f"Melhor Fitness Geral Alcançado: {best_fitness_overall:.2f}")
    
    if best_weights_overall is not None:
        np.save("best_bat_weights.npy", best_weights_overall)
        print("Melhores pesos salvos em 'best_bat_weights.npy'")
        
        # Plota evolução
        plot_evolution(evolution_scores)
        
        # Testa o agente
        test_agent(best_weights_overall, num_tests=30, render=True)
    else:
        print("Nenhum peso ótimo encontrado.")

def plot_evolution(scores):
    """Plota o gráfico da evolução do agente"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(scores)+1), scores, 'b-', linewidth=2)
    plt.xlabel("Geração")
    plt.ylabel("Melhor Fitness")
    plt.title("Evolução do Agente - Algoritmo dos Morcegos (Bat Algorithm)")
    plt.grid(True, alpha=0.3)
    plt.savefig("bat_evolution.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Gráfico salvo como 'bat_evolution.png'")

if __name__ == "__main__":
    train_and_test()
