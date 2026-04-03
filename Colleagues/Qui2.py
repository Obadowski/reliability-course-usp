import numpy as np
from scipy import stats

# ==========================================
# 1. FUNÇÃO PRINCIPAL DO TESTE QUI-QUADRADO
# ==========================================
def teste_qui_quadrado(frequencias_observadas, limites_classes, modelo='expon', parametros_conhecidos=None, nivel_confianca=0.95):
    """
    Executa o teste Qui-Quadrado para dados agrupados em intervalos,
    com normalização de frequências e nível de confiança dinâmico.
    """
    freq_obs = np.array(frequencias_observadas)
    total_amostras = np.sum(freq_obs)
    dist = getattr(stats, modelo)
    
    # 1. Calcula as probabilidades Esperadas usando a área sob a curva (CDF)
    probabilidades_esperadas = []
    for i in range(len(limites_classes) - 1):
        limite_inferior = limites_classes[i]
        limite_superior = limites_classes[i+1]
        
        prob_inf = dist.cdf(limite_inferior, *parametros_conhecidos)
        prob_sup = dist.cdf(limite_superior, *parametros_conhecidos)
        probabilidades_esperadas.append(prob_sup - prob_inf)
        
    freq_esperadas = np.array(probabilidades_esperadas) * total_amostras
    
    # 2. Normalização: Garante que a soma das esperadas seja idêntica às observadas
    # Isso evita erros de precisão numérica do SciPy em caudas cortadas (como a Normal no 0)
    soma_esperadas = np.sum(freq_esperadas)
    if soma_esperadas > 0:
        freq_esperadas = freq_esperadas * (total_amostras / soma_esperadas)
    
    # 3. Executa o Teste Qui-Quadrado
    estatistica_chi2, p_valor = stats.chisquare(f_obs=freq_obs, f_exp=freq_esperadas)
    
    # 4. Avaliação baseada no Nível de Confiança
    alpha = 1.0 - nivel_confianca
    aderencia = "Aceito (Segue o modelo)" if p_valor > alpha else "Rejeitado (Não segue)"
    
    return estatistica_chi2, p_valor, aderencia, freq_esperadas, alpha

# ==========================================
# 2. INTERFACE DO USUÁRIO (INPUTS)
# ==========================================
print("\n=== TESTE DE ADERÊNCIA QUI-QUADRADO (X²) ===")
try:
    # Frequências
    entrada_obs = input("Digite as frequências observadas separadas por ESPAÇO: ").replace(',', '.')
    obs_user = [float(valor) for valor in entrada_obs.split()]
    
    # Limites
    entrada_limites = input("Digite os limites de tempo (containers ou caixas) e sempre use 'inf' para infinito ao final - Ex: 0 100 200 inf: ").replace(',', '.')
    limites_user = [np.inf if valor.lower() == 'inf' else float(valor) for valor in entrada_limites.split()]

    # Trava de segurança para garantir a regra das caixas
    if len(limites_user) != len(obs_user) + 1:
        print("\n[!] AVISO: A quantidade de limites digitados não bate com as frequências.")
        print(f"Você digitou {len(obs_user)} frequências, então precisa digitar {len(obs_user) + 1} limites.")
    else:
        # Define o modelo
        print("\nModelos disponíveis:")
        print("- 'norm'           (Normal/Gaussiana)")
        print("- 'expon'          (Exponencial)")
        print("- 'uniform'        (Uniforme)")
        print("- 'gamma'          (Gama)")
        print("- 'beta'           (Beta)")
        print("- 'weibull_min'    (Weibull Mínimo)")
        print("- 'weibull_max'    (Weibull Máximo)")
        print("- 'lognorm'        (Log-Normal)")
        print("- 'chi2'           (Qui-Quadrado)")
        print("- 't'              (t de Student)")
        print("- 'f'              (F de Snedecor)")
        print("- 'laplace'        (Laplace)")
        print("- 'cauchy'         (Cauchy)")
        print("- 'poisson'        (Poisson)")
        print("- 'binom'          (Binomial - Quantos sucessos em n tentativas?)") 
        print("- 'nbinom'         (Binomial - Quantas tentativas até n sucessos?)")
    
        modelo_user = input("Digite a sigla do modelo: ").strip().lower()
        
        # Nível de Confiança
        entrada_conf = input("Digite o nível de confiança em % (Ex: 95): ").replace(',', '.')
        val_conf = float(entrada_conf)
        confianca_user = val_conf / 100.0 if val_conf > 1 else val_conf

        # Instruções de preenchimento
        print("\n[Instruções de Parâmetros]")
        print("- 'norm':        média e desvio padrão. Ex: 100 15")
        print("- 'expon':       escala (1/lambda). Ex: 2000")
        print("- 'uniform':     loc (mínimo) e scale (máximo - mínimo). Ex: 0 100")
        print("- 'gamma':       forma (a), loc (0) e escala. Ex: 2 0 10")
        print("- 'beta':        forma1 (a), forma2 (b), loc (0) e scale (1). Ex: 2 5 0 1")
        print("- 'weibull_min': forma (c), loc (0) e escala (eta). Ex: 2.5 0 1000")
        print("- 'lognorm':     forma (s), loc (0) e escala (mediana). Ex: 0.5 0 100")
        print("- 'chi2':        graus de liberdade (df). Ex: 5")
        print("- 't':           graus de liberdade (df). Ex: 10")
        print("- 'f':           df1 (numerador) e df2 (denominador). Ex: 5 10")
        print("- 'laplace':     loc (média) e scale (desvio). Ex: 0 1")
        print("- 'cauchy':      loc (pico) e scale (escala). Ex: 0 1")
        print("- 'poisson':     mu (taxa média). Ex: 5")
        print("- 'binom':       n (tentativas) e p (probabilidade). Ex: 100 0.5")
        print("- 'geom':        p (probabilidade de sucesso). Ex: 0.3")
        print("- 'nbinom':      n (sucessos desejados) e p (probabilidade). Ex: 5 0.5")
        
        entrada_param = input("\nDigite os parâmetros separados por espaço: ").replace(',', '.')
        parametros_user = tuple(float(valor) for valor in entrada_param.split())
        
        print(f"\nCalculando teste para '{modelo_user}'...")
        
        # Chamada da Função
        chi2, p_val, resultado, f_exp, alpha_calc = teste_qui_quadrado(
            obs_user, limites_user, modelo_user, parametros_user, confianca_user
        )
        
        # Resultados
        print("-" * 65)
        print(f"Frequências Esperadas:   {[round(f, 2) for f in f_exp]}")
        print(f"Estatística Qui-Quadrado: {chi2:.4f}")
        print(f"P-Valor:                 {p_val:.4f}")
        print(f"Nível de Significância (\u03B1): {alpha_calc:.3f} ({alpha_calc*100:.1f}%)")
        print(f"Decisão ({confianca_user*100:.1f}% Confiança): {resultado}")
        print("-" * 65)

except Exception as e:
    print(f"\nErro inesperado: {e}")