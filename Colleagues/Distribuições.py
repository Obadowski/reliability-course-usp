import numpy as np
from scipy import stats

print("\n=== CALCULADORA UNIVERSAL DE DISTRIBUIÇÕES ===")
try:
    print("Qual o tipo de variável que você está analisando?")
    print("1 - DISCRETA  (Eventos contáveis: falhas, peças defeituosas, acidentes)")
    print("2 - CONTÍNUA  (Tempo até falhar, desgaste, medidas contínuas)")
    tipo_var = input("Escolha (1 ou 2): ").strip()

    # ==========================================
    # FLUXO 1: VARIÁVEIS DISCRETAS
    # ==========================================
    if tipo_var == '1':
        print("\n[Modelos Discretos]")
        print("- 'binom'   (Binomial)")
        print("- 'poisson' (Poisson)")
        print("- 'hyper'   (Hipergeométrica)")
        modelo = input("Sigla do modelo: ").strip().lower()

        if modelo == 'binom':
            n = int(input("Número de tentativas (n): "))
            p = float(input("Probabilidade de sucesso/falha (p): ").replace(',', '.'))
            dist = stats.binom(n, p)
            
        elif modelo == 'poisson':
            print("\n[Parâmetros de Poisson]")
            print("1 - Inserir Taxa Média (\u03bb) diretamente")
            print("2 - Calcular a partir de histórico (\u03bb = (Ocorrências / Base) * Alvo)")
            tipo_poisson = input("Escolha (1 ou 2): ").strip()
            
            if tipo_poisson == '1':
                lamb = float(input("Taxa média de ocorrências (\u03bb): ").replace(',', '.'))
            elif tipo_poisson == '2':
                ocorr_base = float(input("Ocorrências observadas no passado: ").replace(',', '.'))
                t_base = float(input("Tamanho do intervalo BASE: ").replace(',', '.'))
                t_alvo = float(input("Tamanho do intervalo ALVO da pergunta: ").replace(',', '.'))
                lamb = (ocorr_base / t_base) * t_alvo
            else:
                raise ValueError("Opção inválida.")
            dist = stats.poisson(lamb)
            
        elif modelo == 'hyper':
            M = int(input("População TOTAL (N): "))
            n_def = int(input("Total de itens com a característica na população (K): "))
            N_am = int(input("Tamanho da amostra (n): "))
            dist = stats.hypergeom(M, n_def, N_am)
        else:
            raise ValueError("Modelo discreto não reconhecido.")

        x = int(input("\nQual a quantidade alvo de ocorrências (x)? (ex: 2): "))
        
        prob_exata = dist.pmf(x)
        prob_acumulada = dist.cdf(x)
        prob_maior = dist.sf(x)

        print("-" * 55)
        print(f"P(X = {x})  [Exatamente {x}]: {prob_exata:.6f} ({prob_exata*100:.2f}%)")
        print(f"P(X <= {x}) [No máximo {x}]:  {prob_acumulada:.6f} ({prob_acumulada*100:.2f}%)")
        print(f"P(X > {x})  [Mais que {x}]:   {prob_maior:.6f} ({prob_maior*100:.2f}%)")
        print(f"P(X >= {x}) [Pelo menos {x}]: {(prob_maior + prob_exata):.6f} ({(prob_maior + prob_exata)*100:.2f}%)")
        print("-" * 55)

    # ==========================================
    # FLUXO 2: VARIÁVEIS CONTÍNUAS
    # ==========================================
    elif tipo_var == '2':
        print("\n[Modelos Contínuos]")
        print("- 'norm'        (Normal)")
        print("- 'expon'       (Exponencial)")
        print("- 'weibull_min' (Weibull Padrão/Mínima)")
        print("- 'weibull_max' (Weibull Máxima)")
        print("- 'lognorm'     (Log-Normal)")
        print("- 'gamma'       (Gama)")
        print("- 'beta'        (Beta)")
        modelo = input("Sigla do modelo: ").strip().lower()

        correcao = 0.0

        if modelo == 'norm':
            print("\n[Parâmetros da Normal]")
            print("1 - Inserir Média (\u03bc) e Desvio Padrão (\u03c3)")
            print("2 - Usar Aproximação da Binomial (informar n e p)")
            print("3 - Calcular a partir de uma Amostra de Dados brutos")
            print("4 - Engenharia Reversa (Descobrir \u03bc e \u03c3 a partir de 2 pontos)")
            tipo_norm = input("Escolha (1, 2, 3 ou 4): ").strip()
            
            if tipo_norm == '1':
                mu = float(input("Média (\u03bc): ").replace(',', '.'))
                sigma = float(input("Desvio padrão (\u03c3): ").replace(',', '.'))
            elif tipo_norm == '2':
                n = int(input("Tamanho da amostra (n): "))
                p = float(input("Probabilidade (p): ").replace(',', '.'))
                mu = n * p
                sigma = np.sqrt(n * p * (1 - p))
                corr_inp = input("Aplicar Correção de Continuidade (+/- 0.5)? (s/n): ").strip().lower()
                correcao = 0.5 if corr_inp == 's' else 0.0
            elif tipo_norm == '3':
                entrada_dados = input("Digite os dados amostrais separados por ESPAÇO: ").replace(',', '.')
                dados = np.array([float(v) for v in entrada_dados.split()])
                mu = np.mean(dados)
                sigma = np.std(dados, ddof=1)
            elif tipo_norm == '4':
                print("\n[Insira os dois pontos conhecidos e suas ÁREAS À ESQUERDA]")
                x1 = float(input("Ponto 1 (x1): ").replace(',', '.'))
                p1 = float(input(f"Probabilidade acumulada até {x1} (ex: 0.05 para 5%): ").replace(',', '.'))
                x2 = float(input("Ponto 2 (x2): ").replace(',', '.'))
                p2 = float(input(f"Probabilidade acumulada até {x2} (ex: 0.95 para 95%): ").replace(',', '.'))
                
                # Conversor de segurança caso o usuário digite 5 em vez de 0.05
                if p1 > 1: p1 /= 100.0
                if p2 > 1: p2 /= 100.0
                
                z1 = stats.norm.ppf(p1)
                z2 = stats.norm.ppf(p2)
                sigma = (x2 - x1) / (z2 - z1)
                mu = x1 - (z1 * sigma)
                print(f"\n[Cálculo Interno] Z1 = {z1:.4f} | Z2 = {z2:.4f}")
            else:
                raise ValueError("Opção inválida.")
                
            print("\n[RESUMO DOS PARÂMETROS - NORMAL]")
            print(f" -> Média (\u03bc):          {mu:.4f}")
            print(f" -> Desvio Padrão (\u03c3): {sigma:.4f}")
            dist = stats.norm(loc=mu, scale=sigma)
            
        elif modelo == 'expon':
            print("\n[Parâmetros da Exponencial]")
            print("1 - Inserir Taxa de Falha (\u03bb) diretamente")
            print("2 - Inserir Escala / MTTF (1/\u03bb) diretamente")
            print("3 - Calcular a partir de falhas e tempo (\u03bb = \u03c1 / t)")
            tipo_expon = input("Escolha (1, 2 ou 3): ").strip()
            
            if tipo_expon == '1':
                lamb = float(input("Taxa de falha (\u03bb): ").replace(',', '.'))
                escala = 1.0 / lamb if lamb != 0 else float('inf')
            elif tipo_expon == '2':
                escala = float(input("Escala (MTTF): ").replace(',', '.'))
            elif tipo_expon == '3':
                ro = float(input("Número total de falhas (\u03c1): ").replace(',', '.'))
                t_total = float(input("Tempo total de teste (t): ").replace(',', '.'))
                lamb = ro / t_total if t_total != 0 else 0.0
                escala = 1.0 / lamb if lamb != 0 else float('inf')
            else:
                raise ValueError("Opção inválida.")
            dist = stats.expon(loc=0, scale=escala)
            
        elif modelo == 'weibull_min':
            forma = float(input("Parâmetro de Forma (\u03b2): ").replace(',', '.'))
            escala = float(input("Parâmetro de Escala (\u03b7): ").replace(',', '.'))
            dist = stats.weibull_min(c=forma, loc=0, scale=escala)
            
        elif modelo == 'weibull_max':
            forma = float(input("Parâmetro de Forma (c): ").replace(',', '.'))
            escala = float(input("Parâmetro de Escala: ").replace(',', '.'))
            dist = stats.weibull_max(c=forma, loc=0, scale=escala)
            
        elif modelo == 'lognorm':
            mu_log = float(input("Média do log natural (\u03bc_log): ").replace(',', '.'))
            sigma_log = float(input("Desvio padrão do log natural (\u03c3_log): ").replace(',', '.'))
            dist = stats.lognorm(s=sigma_log, scale=np.exp(mu_log))
            
        elif modelo == 'gamma':
            forma = float(input("Parâmetro de Forma (k ou \u03b1): ").replace(',', '.'))
            escala = float(input("Parâmetro de Escala (\u03b8): ").replace(',', '.'))
            dist = stats.gamma(a=forma, loc=0, scale=escala)
            
        elif modelo == 'beta':
            alfa = float(input("Parâmetro de Forma 1 (\u03b1): ").replace(',', '.'))
            beta_param = float(input("Parâmetro de Forma 2 (\u03b2): ").replace(',', '.'))
            loc = float(input("Limite Inferior (loc): ").replace(',', '.'))
            amp = float(input("Amplitude (escala): ").replace(',', '.'))
            dist = stats.beta(a=alfa, b=beta_param, loc=loc, scale=amp)
            
        else:
            raise ValueError("Modelo contínuo não reconhecido.")

        print("\nO que você deseja calcular?")
        print("1 - P(X < x)  [Área à esquerda / Acumulada]")
        print("2 - P(X > x)  [Área à direita / Sobrevivência]")
        print("3 - P(x1 < X < x2) [Área entre dois valores]")
        print("4 - Encontrar valor de X a partir de uma probabilidade (Engenharia Reversa)")
        pergunta = input("Escolha (1, 2, 3 ou 4): ").strip()

        print("-" * 55)
        if pergunta == '1':
            x = float(input("Digite o valor de x: ").replace(',', '.'))
            x_calc = x + correcao if correcao else x
            prob = dist.cdf(x_calc)
            print(f"Probabilidade P(X < {x}): {prob:.6f} ({prob*100:.2f}%)")
            
        elif pergunta == '2':
            x = float(input("Digite o valor de x: ").replace(',', '.'))
            x_calc = x - correcao if correcao else x
            prob = dist.sf(x_calc)
            print(f"Probabilidade P(X > {x}): {prob:.6f} ({prob*100:.2f}%)")
            
        elif pergunta == '3':
            x1 = float(input("Digite o limite INFERIOR (x1): ").replace(',', '.'))
            x2 = float(input("Digite o limite SUPERIOR (x2): ").replace(',', '.'))
            x1_calc = x1 - correcao if correcao else x1
            x2_calc = x2 + correcao if correcao else x2
            prob = dist.cdf(x2_calc) - dist.cdf(x1_calc)
            print(f"Probabilidade P({x1} < X < {x2}): {prob:.6f} ({prob*100:.2f}%)")
            
        elif pergunta == '4':
            print("DICA: Sempre insira a probabilidade acumulada (Área à ESQUERDA do valor de X)")
            p_alvo = float(input("Digite a probabilidade [ex: 0.01 para 1%]: ").replace(',', '.'))
            if p_alvo > 1: p_alvo /= 100.0
            x_encontrado = dist.ppf(p_alvo)
            print(f"O valor de X que acumula {p_alvo*100:.2f}% de probabilidade é: {x_encontrado:.4f}")
            
        else:
             print("Opção de cálculo inválida.")
        print("-" * 55)

    else:
        print("Opção inválida. Execute o código novamente e escolha 1 ou 2.")

except Exception as e:
    print(f"\n[!] Erro durante a execução: {e}")