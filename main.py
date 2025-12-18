import itertools

import random
import statistics
from math import ceil

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from prettytable import PrettyTable

from typing import Callable, List, Tuple

class BPP:
    """Represents a Bin Packing Problem (BPP) configuration.

    Attributes:
        num_items (int): Number of items.
        num_bins (int): Number of bins.
        item_weight (Callable[[int], float]): Function to compute item weights.
    """

    def __init__(self, item_weight: Callable[[int], float], num_bins: int, num_items: int = 500) -> None:
        """Initialises a Bin Packing Problem.
        
        Args:
            item_weight (Callable[[int], float]): Function to compute item weights.
            num_bins (int): Number of bins.
            num_items (int, optional): Number of items. Defaults to 500."""
        self.num_items = num_items
        self.num_bins = num_bins
        self.item_weight = item_weight

class GeneticAlgorithm:
    """Implements a Genetic Algorithm (GA) to approximate a solution for the Bin Packing Problem (BPP).

    Attributes:
        bpp (BPP): The bin packing problem instance.
        population_size (int): Size of the GA population.
        pm (float): Mutation probability.
        tournament_size (int): Number of chromosomes in tournament selection.
        pc (float): Crossover probability.
        elitism (int): Number of fittest chromosomes preserved in each generation.
        termination (int): Number of fitness evaluations before termination.
        num_evals (int): Counter for total fitness evaluations.
        avg_fitnesses (list[float]): Record of average fitness per generation.
        max_fitnesses (list[float]): Record of maximum fitness per generation.
    """

    def __init__(self, bpp: BPP, population_size: int, pm: float, tournament_size: int, pc: float = 0.8, elitism: int = 1, termination: int = 10000) -> None:
        """Initialise the genetic algorithm parameters.
        
        Args:
            bpp (BPP): The bin packing problem instance.
            population_size (int): Size of the GA population.
            pm (float): Mutation probability.
            tournament_size (int): Number of chromosomes in tournament selection.
            pc (float, optional): Crossover probability. Default to 0.5.
            elitism (int, optional): Number of fittes chromosomes preserved in each generation. Default to 1.
            termination (int, optional): Number of fitness evaluations before termination. Defaults to 10000."""
        self.bpp = bpp
        self.population_size = population_size
        self.pm = pm
        self.tournament_size = tournament_size
        self.pc = pc
        self.elitism = elitism
        self.termination = termination
        self.num_evals = 0
        self.avg_fitnesses = []
        self.max_fitnesses = []
        
    def _initialise_pop(self) -> List[List[int]]:
        """Generate the initial random population.

        Returns:
            list[list[int]]: A population of chromosomes (each chromosome is a list of bin assignments).
        """
        population = []
        for _ in range(0, self.population_size):
            # generate random chromosome
            population.append([random.randrange(1, self.bpp.num_bins + 1) for _ in range(0, self.bpp.num_items)])
        return population

    def _evaluate_fitnesses(self, population: List[List[int]]) -> List[float]:
        """Evaluate the fitness of each chromosome in the population.

        Fitness is inversely proportional to the weight difference between the heaviest and lightest bins.

        Args:
            population (list[list[int]]): Current generation of chromosomes.

        Returns:
            list[float]: Fitness values for each chromosome.
        """
        fitnesses = []
        for chromosome in population:
            bin_difference = self._weight_difference(chromosome)
            fitness = 100 / (1 + bin_difference) # use fitness function as defined in specification
            fitnesses.append(fitness)

        # increment termination counter by number of fitness evaluations
        self.num_evals += self.population_size

        return fitnesses
    
    def _weight_difference(self, chromosome: List[int]) -> float:
        """Compute the difference between the heaviest and lightest bins.

        Args:
            chromosome (list[int]): A single chromosome representing item-to-bin assignments.

        Returns:
            float: The weight difference between the heaviest and lightest bins.
        """
        bin_weights = {i: 0 for i in range(1, self.bpp.num_bins + 1)} # generate dict{ bin : bin_weight }
        # for each (index, value) in chromosome
        for i in enumerate(chromosome):
            w = self.bpp.item_weight(i[0]+1) # calculate item weight for index
            bin_weights[i[1]] += w # update bin weight by item weight amount

        # use iterable max & min functions to find heaviest & lightest bins
        max_weight = bin_weights[max(bin_weights, key=bin_weights.get)]
        min_weight = bin_weights[min(bin_weights, key=bin_weights.get)]

        return max_weight - min_weight

    def _select_parents(self, population: List[List[int]], fitnesses: List[float]) -> List[List[int]]:
        """Select two parents using tournament selection.

        Args:
            population (list[list[int]]): Current generation of chromosomes.
            fitnesses (list[float]): Corresponding fitness values.

        Returns:
            list[list[int]]: Two parent chromosomes.
        """
        parents = []
        # repeat twice to get 2 parents
        for _ in range(2):
            # randomly select t chromosomes from population
            tournament_i = random.sample(range(0, self.population_size), self.tournament_size)

            tournament = [population[i] for i in tournament_i] # tournament
            tournament_fit = [fitnesses[i] for i in tournament_i] # tournament fitnesses
            
            # append tournament member with best fitness
            parents.append(tournament[tournament_fit.index(max(tournament_fit))])

        return parents

    def _crossover(self, parents: List[List[int]]) -> List[List[int]]:
        """Perform uniform crossover on two parent chromosomes.

        Args:
            parents (list[list[int]]): Two parent chromosomes.

        Returns:
            list[list[int]]: Two offspring chromosomes.
        """
        parent1, parent2 = parents
        # for each gene position, take from either parent 1 or 2 at 50/50 probability
        mapping = [random.choice([0, 1]) for _ in range(self.bpp.num_items)] # binary mapping for element-wise crossover
        child1 = [parent1[i] if mapping[i] == 0 else parent2[i] for i in range(self.bpp.num_items)]
        child2 = [parent2[i] if mapping[i] == 0 else parent1[i] for i in range(self.bpp.num_items)]

        return [child1, child2]

    def _mutate(self, chromosome: List[int]) -> List[int]:
        """Apply random reassignment mutation to children.

        Args:
            chromosome (list[int]): Chromosome to mutate.

        Returns:
            list[int]: Mutated chromosome.
        """
        mutation = chromosome.copy()
        # for each gene in chromosome
        for i in range(self.bpp.num_items):
            if random.random() < self.pm: # at probability pm
                mutation[i] = random.randrange(1, self.bpp.num_bins + 1) # replace bin assignment to any valid bin (1 - b)

        return mutation
    
    def _get_stats(self, population: List[List[int]], fitnesses: List[float]) -> Tuple[float, float, float]:
        """Compute statistics from the final population.

        Args:
            population (list[list[int]]): Final generation of chromosomes.
            fitnesses (list[float]): Corresponding fitness values.

        Returns:
            tuple[float, float, float]: Best fitness, fitness percentage, and standard deviation of bin weights.
        """

        best_f = max(fitnesses)
        best = population[fitnesses.index(max(fitnesses))]

        # get list of total weights in bins (index=bin, value=weight)
        bin_weights = [0]*self.bpp.num_bins
        for i in enumerate(best): # for (index, value) in chromosome
            w = self.bpp.item_weight(i[0]+1)
            bin_weights[i[1]-1] += w

        # average weight of bins
        avg_w = statistics.mean(bin_weights)

        # standard deviation of bin weights
        std_w = statistics.stdev(bin_weights)

        # accuracy of fitness
        best_f_perc = max(0, 100*(1-(std_w/avg_w)))

        return best_f, best_f_perc, std_w

    def run(self, rand_seed: int) -> Tuple[float, float, float]:
        """Run the genetic algorithm on the provided BPP instance.

        Args:
            rand_seed (int): Random seed for reproducibility.

        Returns:
            tuple[float, float, float]: Final best fitness, percentage accuracy, and standard deviation of bin weights.
        """

        random.seed(rand_seed)

        # initialise population
        population = self._initialise_pop()

        # initialise records for best and avg fitnesses
        self.avg_fitnesses = []
        self.max_fitnesses = []

        # repeating until termination (i.e., 10000 population fitness evaluations)
        self.num_evals = 0
        while self.num_evals <= self.termination:

            # evaluate fitnesses of population
            fitnesses = self._evaluate_fitnesses(population)

            # record avg & max fitnesses
            self.avg_fitnesses.append(statistics.mean(fitnesses))
            self.max_fitnesses.append(max(fitnesses))

            new_population = []

            # find fittest candidates to keep (amount depends on elitism param)
            elites_fit = sorted(fitnesses, reverse=True)[:self.elitism]
            for f in elites_fit:
                f_i = fitnesses.index(f)
                new_population.append(population[f_i])

            # until whole population is replaced (generational replacement)
            while len(new_population) < self.population_size:

                # --- SELECTION ---
                parents = self._select_parents(population, fitnesses)

                # --- CROSSOVER ---
                if random.random() < self.pc:
                    children = self._crossover(parents)
                else:
                    children = parents # if crossover not applied, use parents

                # --- MUTATION ---
                child1 = self._mutate(children[0])
                child2 = self._mutate(children[1])

                # --- REPLACEMENT ---
                # if we only need one more child to meet population_size, add child1
                if self.population_size - len(new_population) == 1:
                    new_population.append(child1)
                else:
                    new_population.append(child1)
                    new_population.append(child2)

            population = new_population

        return self._get_stats(population, fitnesses)

# both plotting graphs and making tables
def show_results(title: str, configs: List[GeneticAlgorithm], num_trials: int = 5) -> None:
    """Run GA configurations. Display results in tables and plot convergence graphs.

    Args:
        title (str): Title for trials.
        configs (list[GeneticAlgorithm]): List of GA configurations to test.
        num_trials (int, optional): Number of independent runs per configuration. Defaults to 5.
    """

    # intialise table
    table = PrettyTable(['Config', 'Fitness (best)', 'Fitness % (best)', 'Fitness (avg)', 'Std Bin Weight (best)'])
    table.align = 'l'

    # intialise figure for graphs
    fig, axes = plt.subplots(2 if len(configs) > 1 else 1, ceil(len(configs)/2), figsize=(10, 10), squeeze=False, sharey=True)
    axes = axes.flatten()

    for i, config in enumerate(configs):
        ax = axes[i]
        colours = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

        best_fs = []
        best_f_percs = []
        std_ws = []

        # perform each trial
        for trial in range(num_trials):

            # generate random seed based on parameter values
            seed = int(config.tournament_size * 1000 + config.pm * 1000 + trial * 1000)
            # run trial and record results
            best_f, best_f_perc, std_w = config.run(rand_seed=seed)
            best_fs.append(best_f)
            best_f_percs.append(best_f_perc)
            std_ws.append(std_w)
            
            # plot trial fitnesses (best = solid, avg = dashed)
            colour = next(colours)
            ax.plot(config.avg_fitnesses, linestyle='--', color=colour, linewidth=2.0, alpha=0.6)
            ax.plot(config.max_fitnesses, linestyle='-', color=colour, linewidth=2.0, alpha=1)

        # details for graph
        ax.set_title(f'p={config.population_size}, pm={config.pm}, t={config.tournament_size}', fontsize=15)
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # details for table
        i = best_fs.index(max(best_fs))
        table.add_row([f'p={config.population_size}, pm={config.pm}, t={config.tournament_size}', best_fs[i], best_f_percs[i], statistics.mean(best_fs), std_ws[i]])
    
    # set figure labels, title and legend, then save
    fig.text(0.5, 0.04, 'Generation', ha='center', fontsize=15)
    fig.text(0.04, 0.5, 'Fitness', va='center', rotation='vertical', fontsize=15)
    fig.suptitle(f'{title}: Best Fitness of Generations Over {num_trials} Trials\n', fontsize=16)
    plt.tight_layout(rect=[0.06, 0.05, 1, 1])
    custom_lines = [
        Line2D([0], [0], color='black', linestyle='-', label='Best fitness'),
        Line2D([0], [0], color='black', linestyle='--', label='Average fitness')
    ]
    ax = axes[0]
    ax.legend(handles=custom_lines, loc='upper left')
    plt.savefig(f'./plots/{title}.png', dpi=300)
    
    # show table
    print(f'|--- {title} ---|')
    print(table)


if __name__ == '__main__':
    
    bpp1 = BPP(item_weight= lambda i : i, num_bins=10, num_items=500)
    bpp2 = BPP(item_weight= lambda i : (i**2)/2, num_bins=50, num_items=500)

    # --- STANDARD EXPERIMENT ---

    config_11 = GeneticAlgorithm(bpp=bpp1, population_size=100, pm=0.01, tournament_size=3)
    config_12 = GeneticAlgorithm(bpp=bpp1, population_size=100, pm=0.05, tournament_size=3)
    config_13 = GeneticAlgorithm(bpp=bpp1, population_size=100, pm=0.01, tournament_size=7)
    config_14 = GeneticAlgorithm(bpp=bpp1, population_size=100, pm=0.05, tournament_size=7)

    config_21 = GeneticAlgorithm(bpp=bpp2, population_size=100, pm=0.01, tournament_size=3)
    config_22 = GeneticAlgorithm(bpp=bpp2, population_size=100, pm=0.05, tournament_size=3)
    config_23 = GeneticAlgorithm(bpp=bpp2, population_size=100, pm=0.01, tournament_size=7)
    config_24 = GeneticAlgorithm(bpp=bpp2, population_size=100, pm=0.05, tournament_size=7)

    show_results('BPP1', [config_11, config_12, config_13, config_14])
    show_results('BPP2', [config_21, config_22, config_23, config_24])

    # --- FURTHER EXPERIMENT ---

    config_15 = GeneticAlgorithm(bpp=bpp1, population_size=100, pm=0.001, tournament_size=10, elitism=3)
    config_25 = GeneticAlgorithm(bpp=bpp2, population_size=100, pm=0.001, tournament_size=10, elitism=3)

    show_results('BPP1_Further', [config_15])
    show_results('BPP2_Further', [config_25])

# TODO: type hinting