import numpy as np
from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent, BatAlgorithmTrainer
import matplotlib.pyplot as plt
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

POPULATION_SIZE = 100
MAX_ITERATIONS = 1000
MAX_TIME = 12 * 3600  # 12 horas em segundos

def game_fitness_function(weights: np.ndarray) -> float:
    """Avalia um conjunto de pesos da rede neural através de 3 execuções do jogo"""
    scores = []
    for _ in range(3):
        game_config = GameConfig(num_players=1)
        game = SurvivalGame(config=game_config, render=False)
        agent = NeuralNetworkAgent(weights)
        
        while not game.all_players_dead():
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
            
        scores.append(game.players[0].score)
    
    return np.mean(scores)

def train_bat_algorithm():
    print("\n--- Iniciando Treinamento com Bat Algorithm ---")
    
    # Dimensão dos pesos da rede neural: 28*32 + 33*16 + 17*3
    weight_dim = 28*32 + 33*16 + 17*3  # 1379 pesos totais
    
    trainer = BatAlgorithmTrainer(
        pop_size=POPULATION_SIZE,
        num_iterations=1,
        game_eval_fn=game_fitness_function,
        weight_dim=weight_dim,
        fmin=0,
        fmax=2,
        alpha=0.9,
        gamma=0.9
    )
    
    best_scores = []
    best_bat = None
    best_score = -np.inf
    
    start_time = time.time()
    
    for iteration in range(MAX_ITERATIONS):
        # Verifica limite de tempo
        if (time.time() - start_time) > MAX_TIME:
            print("Interrompido por tempo (12h).")
            break
            
        start_iteration = time.time()
        candidate, score = trainer.optimize()
        
        if score > best_score:
            best_score = score
            best_bat = candidate.copy()
            # Salva backup do melhor
            np.save("best_bat_weights.npy", best_bat)
        
        best_scores.append(best_score)
        end_iteration = time.time()
        
        print(f"{iteration + 1}/{MAX_ITERATIONS} | Melhor Score: {best_score:.2f} | ({end_iteration-start_iteration:.2f}s)")
    
    print("\n--- Treinamento Concluído ---")
    print(f"Melhor Score Alcançado: {best_score:.2f}")
    
    if best_bat is not None:
        np.save("best_bat_weights.npy", best_bat)
        print("Melhores pesos salvos em 'best_bat_weights.npy'")
    
    return best_scores, best_bat

def plot_evolution(scores):
    """Plota o gráfico da evolução do agente"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(scores)+1), scores, 'b-', linewidth=2)
    plt.xlabel("Iteração")
    plt.ylabel("Melhor Pontuação")
    plt.title("Evolução do Agente - Algoritmo dos Morcegos (Bat Algorithm)")
    plt.grid(True, alpha=0.3)
    plt.savefig("bat_evolution.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Gráfico salvo como 'bat_evolution.png'")

if __name__ == "__main__":
    # Treinamento
    scores, best_bat = train_bat_algorithm()
    
    # Plota evolução
    if scores:
        plot_evolution(scores)
    
    # Teste rápido do melhor agente
    if best_bat is not None:
        from test_trained_agent_bat import test_agent
        test_agent(best_bat, num_tests=3, render=False)
