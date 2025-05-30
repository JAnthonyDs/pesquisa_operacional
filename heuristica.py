import random
from typing import List, Dict, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import time
import psutil
import os

@dataclass
class Processo:
    id: int
    categoria: int
    peso: float
    urgencia: int  # Nível de urgência (1-5)
    tempo_estimado: float

@dataclass
class Funcionario:
    id: int
    especialidades: Set[int]
    carga_horaria: float
    senioridade: int  # Nível de senioridade (1-5)
    carga_atual: float = 0.0

class AlgoritmoGenetico:
    def __init__(self, 
                 num_geracoes: int = 100,
                 tamanho_populacao: int = 50,
                 taxa_mutacao: float = 0.1,
                 taxa_crossover: float = 0.8):
        self.num_geracoes = num_geracoes
        self.tamanho_populacao = tamanho_populacao
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.funcionarios: List[Funcionario] = []
        self.processos: List[Processo] = []
        self.melhor_solucao = None
        self.melhor_fitness = float('inf')
        self.tempo_inicio = 0
        self.memoria_inicial = 0

    def iniciar_metricas(self):
        """Inicia o monitoramento de métricas."""
        self.tempo_inicio = time.time()
        processo = psutil.Process(os.getpid())
        self.memoria_inicial = processo.memory_info().rss

    def finalizar_metricas(self):
        """Finaliza o monitoramento e retorna as métricas."""
        tempo_total = time.time() - self.tempo_inicio
        processo = psutil.Process(os.getpid())
        memoria_final = processo.memory_info().rss
        memoria_usada = memoria_final - self.memoria_inicial
        return tempo_total, memoria_usada

    def adicionar_funcionario(self, funcionario: Funcionario):
        self.funcionarios.append(funcionario)

    def adicionar_processo(self, processo: Processo):
        self.processos.append(processo)

    def gerar_solucao_aleatoria(self) -> Dict[int, List[int]]:
        """Gera uma solução aleatória válida."""
        solucao = defaultdict(list)
        funcionarios_disponiveis = self.funcionarios.copy()
        
        # Ordena processos por urgência (mais urgentes primeiro)
        processos_ordenados = sorted(self.processos, key=lambda p: -p.urgencia)
        
        for processo in processos_ordenados:
            # Encontra funcionários compatíveis
            funcionarios_compativeis = [
                f for f in funcionarios_disponiveis 
                if processo.categoria in f.especialidades
            ]
            
            if not funcionarios_compativeis:
                continue
                
            # Escolhe funcionário aleatoriamente entre os compatíveis
            funcionario = random.choice(funcionarios_compativeis)
            solucao[funcionario.id].append(processo.id)
            
        return dict(solucao)

    def calcular_fitness(self, solucao: Dict[int, List[int]]) -> float:
        """Calcula o fitness de uma solução."""
        # Reinicia cargas dos funcionários
        for f in self.funcionarios:
            f.carga_atual = 0.0
        
        # Calcula cargas
        for func_id, processos_ids in solucao.items():
            funcionario = next(f for f in self.funcionarios if f.id == func_id)
            for proc_id in processos_ids:
                processo = next(p for p in self.processos if p.id == proc_id)
                funcionario.carga_atual += processo.peso
        
        # Calcula diferença máxima de carga
        cargas = [f.carga_atual for f in self.funcionarios]
        diferenca_maxima = max(cargas) - min(cargas)
        
        # Penaliza soluções inválidas
        penalidade = 0
        for f in self.funcionarios:
            if f.carga_atual > f.carga_horaria:
                penalidade += (f.carga_atual - f.carga_horaria) * 1000
        
        return diferenca_maxima + penalidade

    def crossover(self, pai1: Dict[int, List[int]], pai2: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """Realiza o crossover entre duas soluções."""
        if random.random() > self.taxa_crossover:
            return pai1.copy()
        
        filho = defaultdict(list)
        todos_processos = set()
        
        # Coleta todos os processos dos pais
        for processos in pai1.values():
            todos_processos.update(processos)
        for processos in pai2.values():
            todos_processos.update(processos)
        
        # Distribui processos aleatoriamente
        for processo_id in todos_processos:
            funcionarios_compativeis = [
                f for f in self.funcionarios 
                if self.processos[processo_id].categoria in f.especialidades
            ]
            
            if funcionarios_compativeis:
                funcionario = random.choice(funcionarios_compativeis)
                filho[funcionario.id].append(processo_id)
        
        return dict(filho)

    def mutacao(self, solucao: Dict[int, List[int]]):
        """Aplica mutação em uma solução."""
        if random.random() > self.taxa_mutacao:
            return
        
        # Escolhe um processo aleatoriamente
        todos_processos = []
        for processos in solucao.values():
            todos_processos.extend(processos)
        
        if not todos_processos:
            return
            
        processo_id = random.choice(todos_processos)
        processo = self.processos[processo_id]
        
        # Remove o processo de sua atribuição atual
        for func_id in solucao:
            if processo_id in solucao[func_id]:
                solucao[func_id].remove(processo_id)
                break
        
        # Reatribui o processo
        funcionarios_compativeis = [
            f for f in self.funcionarios 
            if processo.categoria in f.especialidades
        ]
        
        if funcionarios_compativeis:
            funcionario = random.choice(funcionarios_compativeis)
            solucao[funcionario.id].append(processo_id)

    def executar(self) -> Dict[int, List[int]]:
        """Executa o algoritmo genético."""
        self.iniciar_metricas()
        
        # Gera população inicial
        populacao = [self.gerar_solucao_aleatoria() for _ in range(self.tamanho_populacao)]
        
        for geracao in range(self.num_geracoes):
            # Calcula fitness para cada solução
            fitness_populacao = [(sol, self.calcular_fitness(sol)) for sol in populacao]
            fitness_populacao.sort(key=lambda x: x[1])
            
            # Atualiza melhor solução
            if fitness_populacao[0][1] < self.melhor_fitness:
                self.melhor_solucao = fitness_populacao[0][0]
                self.melhor_fitness = fitness_populacao[0][1]
            
            # Seleciona melhores soluções
            melhores_solucoes = [sol for sol, _ in fitness_populacao[:self.tamanho_populacao//2]]
            
            # Gera nova população
            nova_populacao = melhores_solucoes.copy()
            
            while len(nova_populacao) < self.tamanho_populacao:
                pai1 = random.choice(melhores_solucoes)
                pai2 = random.choice(melhores_solucoes)
                filho = self.crossover(pai1, pai2)
                self.mutacao(filho)
                nova_populacao.append(filho)
            
            populacao = nova_populacao
            
            if geracao % 10 == 0:
                print(f"Geração {geracao}: Melhor fitness = {self.melhor_fitness}")
        
        tempo, memoria = self.finalizar_metricas()
        print(f"\nMétricas de Execução:")
        print(f"Tempo de execução: {tempo:.4f} segundos")
        print(f"Memória utilizada: {memoria / 1024 / 1024:.2f} MB")
        
        return self.melhor_solucao

def gerar_dados_teste():
    # Gerar funcionários
    funcionarios = [
        Funcionario(1, {1, 2}, 44.0, senioridade=3),    # Especialista em categorias 1 e 2
        Funcionario(2, {2, 3}, 40.0, senioridade=4),    # Especialista em categorias 2 e 3
        Funcionario(3, {3, 4}, 36.0, senioridade=2),    # Especialista em categorias 3 e 4
        Funcionario(4, {4, 5}, 44.0, senioridade=5),    # Especialista em categorias 4 e 5
        Funcionario(5, {1, 5}, 40.0, senioridade=3),    # Especialista em categorias 1 e 5
        Funcionario(6, {1, 3}, 44.0, senioridade=4),    # Especialista em categorias 1 e 3
        Funcionario(7, {2, 4}, 36.0, senioridade=3),    # Especialista em categorias 2 e 4
        Funcionario(8, {3, 5}, 40.0, senioridade=2),    # Especialista em categorias 3 e 5
        Funcionario(9, {1, 4}, 44.0, senioridade=5)     # Especialista em categorias 1 e 4
    ]
    
    # Gerar processos
    processos = []
    for i in range(100):
        categoria = random.randint(1, 5)
        urgencia = random.randint(1, 5)
        tempo = random.uniform(0.5, 5.0)
        peso = (urgencia * 10) / tempo
        processos.append(Processo(
            id=i,
            categoria=categoria,
            peso=peso,
            urgencia=urgencia,
            tempo_estimado=tempo
        ))
    
    return funcionarios, processos

def main():
    # Criar instância do algoritmo genético
    ag = AlgoritmoGenetico()
    
    # Gerar dados de teste
    funcionarios, processos = gerar_dados_teste()
    
    # Adicionar funcionários e processos
    for funcionario in funcionarios:
        ag.adicionar_funcionario(funcionario)
    
    for processo in processos:
        ag.adicionar_processo(processo)
    
    # Executar algoritmo
    melhor_solucao = ag.executar()
    
    # Imprimir resultados
    print("\nMelhor solução encontrada:")
    for funcionario_id, processos_ids in melhor_solucao.items():
        funcionario = next(f for f in funcionarios if f.id == funcionario_id)
        print(f"\nFuncionário {funcionario_id} (Senioridade: {funcionario.senioridade}):")
        print(f"Total de processos: {len(processos_ids)}")
        carga_total = sum(processos[pid].peso for pid in processos_ids)
        print(f"Carga total: {carga_total:.2f}")
        print("Processos:")
        for pid in processos_ids:
            p = processos[pid]
            print(f"  - Processo {pid}: Categoria {p.categoria}, "
                  f"Urgência {p.urgencia}, Peso {p.peso:.2f}")

if __name__ == "__main__":
    main()
