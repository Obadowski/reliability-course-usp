import numpy as np

print("\n=== CADEIAS DE MARKOV (TRANSIÇÃO DE ESTADOS) ===")
try:
    num_estados = int(input("Quantos estados possíveis o sistema tem? (ex: 3 para Bom/Médio/Ruim): "))
    
    print("\n[Vetor de Estado Inicial]")
    entrada_v0 = input(f"Digite as {num_estados} probabilidades iniciais (ex: 1 0 0 se for 100% novo): ").replace(',', '.')
    v0 = np.array([float(v) for v in entrada_v0.split()])
    
    print("\n[Matriz de Transição]")
    print("Digite as probabilidades de transição de cada linha (separadas por espaço):")
    matriz_p = []
    for i in range(num_estados):
        linha = input(f"Linha {i+1} (Estado {i+1} para os demais): ").replace(',', '.')
        matriz_p.append([float(v) for v in linha.split()])
        
    matriz_p = np.array(matriz_p)
    
    passos = int(input("\nQuantos passos no tempo deseja calcular? (ex: 4 trimestres): "))
    
    # Eleva a matriz de transição à potência de n passos
    matriz_pn = np.linalg.matrix_power(matriz_p, passos)
    
    # Multiplica o vetor inicial pela matriz elevada
    v_final = np.dot(v0, matriz_pn)
    
    print("-" * 50)
    print(f"Vetor de Probabilidades após {passos} passo(s):")
    for i, p in enumerate(v_final):
        print(f"Estado {i+1}: {p:.4f} ({p*100:.2f}%)")
    print("-" * 50)

except Exception as e:
    print(f"\nErro ao processar a matriz: {e}")