import numpy as np
from bat_algorithm import BatAlgorithm

def sphere_function(x):
    """Função esfera - mínimo global em x=0"""
    return -np.sum(x**2)

def rastrigin_function(x):
    """Função Rastrigin - múltiplos mínimos locais"""
    n = len(x)
    return -(10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x)))

def test_bat_algorithm():
    """Testa BA em função matemática simples"""
    print("Testando Bat Algorithm na função esfera...")
    
    ba = BatAlgorithm(population_size=50, weight_dim=10, 
                     fmin=0, fmax=2, alpha=0.9, gamma=0.9)
    
    best_fitness = -np.inf
    for generation in range(100):
        best_weights, current_fitness = ba.evolve(sphere_function)
        
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            
        if generation % 20 == 0:
            print(f"Geração {generation}: Melhor fitness = {best_fitness:.6f}")
    
    print(f"Resultado final: {best_weights[:5]}... (fitness: {best_fitness:.6f})")
    print("Esperado: valores próximos de 0")

if __name__ == "__main__":
    test_bat_algorithm()