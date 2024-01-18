# type: ignore
"""modded_pygad_ga.py.

This module provides a modified version of the Genetic Algorithm (GA) class from the PyGAD library.

Classes
-------
ModGA:
    A modified Genetic Algorithm class that extends the functionality of the GA class in the PyGAD library.
"""
import time

import numpy
from pygad import GA


class ModGA(GA):
    """A modified Genetic Algorithm class that extends the functionality of the GA class
    in the PyGAD library.

    Methods
    -------
    run_single_generation() -> None:
        Run a single generation of the genetic algorithm.

    run() -> None:
        Runs the genetic algorithm. This is the main method in which the genetic algorithm is evolved through a
        number of generations.
    """

    def run_single_generation(self) -> None:
        """Run a single generation of the genetic algorithm."""
        num_generations = self.num_generations
        self.num_generations = 1
        self.run()
        self.num_generations = num_generations

    def run(self) -> None:
        """Runs the genetic algorithm.

        This is the main method in which the genetic algorithm is evolved through a
        number of generations.
        """
        if self.valid_parameters is False:
            self.logger.error(
                "Error calling the run() method: \nThe run() method cannot be executed with invalid parameters. "
                "Please check the parameters passed while creating an instance of the GA class.\n"
            )
            raise Exception(
                "Error calling the run() method: \nThe run() method cannot be executed with invalid parameters. "
                "Please check the parameters passed while creating an instance of the GA class.\n"
            )

        # Starting from PyGAD 2.18.0, the 4 properties (best_solutions, best_solutions_fitness, solutions,
        # and solutions_fitness) are no longer reset with each call to the run() method. Instead, they are extended.
        # For example, if there are 50 generations and the user set save_best_solutions=True, then the length of the
        # 2 properties best_solutions and best_solutions_fitness will be 50 after the first call to the run() method,
        # then 100 after the second call, 150 after the third, and so on.

        # self.best_solutions: Holds the best solution in each generation.
        if type(self.best_solutions) is numpy.ndarray:
            self.best_solutions = list(self.best_solutions)
        # self.best_solutions_fitness: A list holding the fitness value of the best solution for each generation.
        if type(self.best_solutions_fitness) is numpy.ndarray:
            self.best_solutions_fitness = list(self.best_solutions_fitness)
        # self.solutions: Holds the solutions in each generation.
        if type(self.solutions) is numpy.ndarray:
            self.solutions = list(self.solutions)
        # self.solutions_fitness: Holds the fitness of the solutions in each generation.
        if type(self.solutions_fitness) is numpy.ndarray:
            self.solutions_fitness = list(self.solutions_fitness)

        if self.on_start is not None:
            self.on_start(self)

        stop_run = False

        # To continue from where we stopped, the first generation index should start from the value of the
        # 'self.generations_completed' parameter.
        if (
            self.generations_completed != 0
            and type(self.generations_completed) in GA.supported_int_types
        ):
            # If the 'self.generations_completed' parameter is not '0', then this means we continue execution.
            generation_first_idx = self.generations_completed
            generation_last_idx = self.num_generations + self.generations_completed
        else:
            # If the 'self.generations_completed' parameter is '0', then stat from scratch.
            generation_first_idx = 0
            generation_last_idx = self.num_generations

        # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness
        # attribute.
        self.last_generation_fitness = self.cal_pop_fitness()

        best_solution, best_solution_fitness, best_match_idx = self.best_solution(
            pop_fitness=self.last_generation_fitness
        )

        # Appending the best solution in the initial population to the best_solutions list.
        if self.save_best_solutions:
            self.best_solutions.append(best_solution)

        for generation in range(generation_first_idx, generation_last_idx):
            if self.on_fitness is not None:
                self.on_fitness(self, self.last_generation_fitness)

            # Appending the fitness value of the best solution in the current generation to the best_solutions_fitness
            # attribute.
            self.best_solutions_fitness.append(best_solution_fitness)

            # Appending the solutions in the current generation to the solutions list.
            if self.save_solutions:
                # self.solutions.extend(self.population.copy())
                population_as_list = self.population.copy()
                population_as_list = [list(item) for item in population_as_list]
                self.solutions.extend(population_as_list)

                self.solutions_fitness.extend(self.last_generation_fitness)

            # Selecting the best parents in the population for mating.
            if callable(self.parent_selection_type):
                (
                    self.last_generation_parents,
                    self.last_generation_parents_indices,
                ) = self.select_parents(
                    self, self.last_generation_fitness, self.num_parents_mating, self
                )
                if type(self.last_generation_parents) is not numpy.ndarray:
                    self.logger.error(
                        "The type of the iterable holding the selected parents is expected to be (numpy.ndarray) but "
                        "{last_generation_parents_type} found.".format(
                            last_generation_parents_type=type(
                                self.last_generation_parents
                            )
                        )
                    )
                    raise TypeError(
                        "The type of the iterable holding the selected parents is expected to be (numpy.ndarray) but "
                        "{last_generation_parents_type} found.".format(
                            last_generation_parents_type=type(
                                self.last_generation_parents
                            )
                        )
                    )
                if type(self.last_generation_parents_indices) is not numpy.ndarray:
                    self.logger.error(
                        "The type of the iterable holding the selected parents' indices is expected to be "
                        "(numpy.ndarray) but {last_generation_parents_indices_type} found.".format(
                            last_generation_parents_indices_type=type(
                                self.last_generation_parents_indices
                            )
                        )
                    )
                    raise TypeError(
                        "The type of the iterable holding the selected parents' indices is expected to be "
                        "(numpy.ndarray) but {last_generation_parents_indices_type} found.".format(
                            last_generation_parents_indices_type=type(
                                self.last_generation_parents_indices
                            )
                        )
                    )
            else:
                (
                    self.last_generation_parents,
                    self.last_generation_parents_indices,
                ) = self.select_parents(
                    self.last_generation_fitness, num_parents=self.num_parents_mating
                )

            # Validate the output of the parent selection step: self.select_parents()
            if self.last_generation_parents.shape != (
                self.num_parents_mating,
                self.num_genes,
            ):
                if self.last_generation_parents.shape[0] != self.num_parents_mating:
                    self.logger.error(
                        "Size mismatch between the size of the selected parents {parents_size_actual} and the expected "
                        "size {parents_size_expected}. It is expected to select ({num_parents_mating}) parents but "
                        "({num_parents_mating_selected}) selected.".format(
                            parents_size_actual=self.last_generation_parents.shape,
                            parents_size_expected=(
                                self.num_parents_mating,
                                self.num_genes,
                            ),
                            num_parents_mating=self.num_parents_mating,
                            num_parents_mating_selected=self.last_generation_parents.shape[
                                0
                            ],
                        )
                    )
                    raise ValueError(
                        "Size mismatch between the size of the selected parents {parents_size_actual} and the expected "
                        "size {parents_size_expected}. It is expected to select ({num_parents_mating}) parents but "
                        "({num_parents_mating_selected}) selected.".format(
                            parents_size_actual=self.last_generation_parents.shape,
                            parents_size_expected=(
                                self.num_parents_mating,
                                self.num_genes,
                            ),
                            num_parents_mating=self.num_parents_mating,
                            num_parents_mating_selected=self.last_generation_parents.shape[
                                0
                            ],
                        )
                    )
                elif self.last_generation_parents.shape[1] != self.num_genes:
                    self.logger.error(
                        "Size mismatch between the size of the selected parents {parents_size_actual} and the expected "
                        "size {parents_size_expected}. Parents are expected to have ({num_genes}) genes but "
                        "({num_genes_selected}) produced.".format(
                            parents_size_actual=self.last_generation_parents.shape,
                            parents_size_expected=(
                                self.num_parents_mating,
                                self.num_genes,
                            ),
                            num_genes=self.num_genes,
                            num_genes_selected=self.last_generation_parents.shape[1],
                        )
                    )
                    raise ValueError(
                        "Size mismatch between the size of the selected parents {parents_size_actual} and the expected "
                        "size {parents_size_expected}. Parents are expected to have ({num_genes}) genes but "
                        "({num_genes_selected}) produced.".format(
                            parents_size_actual=self.last_generation_parents.shape,
                            parents_size_expected=(
                                self.num_parents_mating,
                                self.num_genes,
                            ),
                            num_genes=self.num_genes,
                            num_genes_selected=self.last_generation_parents.shape[1],
                        )
                    )

            if self.last_generation_parents_indices.ndim != 1:
                self.logger.error(
                    "The iterable holding the selected parents indices is expected to have 1 dimension but "
                    "({parents_indices_ndim}) found.".format(
                        parents_indices_ndim=len(self.last_generation_parents_indices)
                    )
                )
                raise ValueError(
                    "The iterable holding the selected parents indices is expected to have 1 dimension but "
                    "({parents_indices_ndim}) found.".format(
                        parents_indices_ndim=len(self.last_generation_parents_indices)
                    )
                )
            elif len(self.last_generation_parents_indices) != self.num_parents_mating:
                self.logger.error(
                    "The iterable holding the selected parents indices is expected to have ({num_parents_mating}) "
                    "values but ({num_parents_mating_selected}) found.".format(
                        num_parents_mating=self.num_parents_mating,
                        num_parents_mating_selected=len(
                            self.last_generation_parents_indices
                        ),
                    )
                )
                raise ValueError(
                    "The iterable holding the selected parents indices is expected to have ({num_parents_mating}) "
                    "values but ({num_parents_mating_selected}) found.".format(
                        num_parents_mating=self.num_parents_mating,
                        num_parents_mating_selected=len(
                            self.last_generation_parents_indices
                        ),
                    )
                )

            if self.on_parents is not None:
                self.on_parents(self, self.last_generation_parents)

            # If self.crossover_type=None, then no crossover is applied and thus no offspring will be created in the
            # next generations. The next generation will use the solutions in the current population.
            if self.crossover_type is None:
                if self.keep_elitism == 0:
                    num_parents_to_keep = (
                        self.num_parents_mating
                        if self.keep_parents == -1
                        else self.keep_parents
                    )
                    if self.num_offspring <= num_parents_to_keep:
                        self.last_generation_offspring_crossover = (
                            self.last_generation_parents[0 : self.num_offspring]
                        )
                    else:
                        self.last_generation_offspring_crossover = numpy.concatenate(
                            (
                                self.last_generation_parents,
                                self.population[
                                    0 : (
                                        self.num_offspring
                                        - self.last_generation_parents.shape[0]
                                    )
                                ],
                            )
                        )
                else:
                    # The steady_state_selection() function is called to select the best solutions (i.e. elitism). The
                    # keep_elitism parameter defines the number of these solutions.
                    # The steady_state_selection() function is still called here even if its output may not be used
                    # given that the condition of the next if statement is True. The reason is that it will be used
                    # later.
                    self.last_generation_elitism, _ = self.steady_state_selection(
                        self.last_generation_fitness, num_parents=self.keep_elitism
                    )
                    if self.num_offspring <= self.keep_elitism:
                        self.last_generation_offspring_crossover = (
                            self.last_generation_parents[0 : self.num_offspring]
                        )
                    else:
                        self.last_generation_offspring_crossover = numpy.concatenate(
                            (
                                self.last_generation_elitism,
                                self.population[
                                    0 : (
                                        self.num_offspring
                                        - self.last_generation_elitism.shape[0]
                                    )
                                ],
                            )
                        )
            else:
                # Generating offspring using crossover.
                if callable(self.crossover_type):
                    self.last_generation_offspring_crossover = self.crossover(
                        self.last_generation_parents,
                        (self.num_offspring, self.num_genes),
                        self,
                    )
                    if (
                        type(self.last_generation_offspring_crossover)
                        is not numpy.ndarray
                    ):
                        self.logger.error(
                            "The output of the crossover step is expected to be of type (numpy.ndarray) but "
                            "{last_generation_offspring_crossover_type} found.".format(
                                last_generation_offspring_crossover_type=type(
                                    self.last_generation_offspring_crossover
                                )
                            )
                        )
                        raise TypeError(
                            "The output of the crossover step is expected to be of type (numpy.ndarray) but "
                            "{last_generation_offspring_crossover_type} found.".format(
                                last_generation_offspring_crossover_type=type(
                                    self.last_generation_offspring_crossover
                                )
                            )
                        )
                else:
                    self.last_generation_offspring_crossover = self.crossover(
                        self.last_generation_parents,
                        offspring_size=(self.num_offspring, self.num_genes),
                    )
                if self.last_generation_offspring_crossover.shape != (
                    self.num_offspring,
                    self.num_genes,
                ):
                    if (
                        self.last_generation_offspring_crossover.shape[0]
                        != self.num_offspring
                    ):
                        self.logger.error(
                            "Size mismatch between the crossover output {crossover_actual_size} and the expected "
                            "crossover output {crossover_expected_size}. It is expected to produce ({num_offspring}) "
                            "offspring but ({num_offspring_produced}) produced.".format(
                                crossover_actual_size=self.last_generation_offspring_crossover.shape,
                                crossover_expected_size=(
                                    self.num_offspring,
                                    self.num_genes,
                                ),
                                num_offspring=self.num_offspring,
                                num_offspring_produced=self.last_generation_offspring_crossover.shape[
                                    0
                                ],
                            )
                        )
                        raise ValueError(
                            "Size mismatch between the crossover output {crossover_actual_size} and the expected "
                            "crossover output {crossover_expected_size}. It is expected to produce ({num_offspring}) "
                            "offspring but ({num_offspring_produced}) produced.".format(
                                crossover_actual_size=self.last_generation_offspring_crossover.shape,
                                crossover_expected_size=(
                                    self.num_offspring,
                                    self.num_genes,
                                ),
                                num_offspring=self.num_offspring,
                                num_offspring_produced=self.last_generation_offspring_crossover.shape[
                                    0
                                ],
                            )
                        )
                    elif (
                        self.last_generation_offspring_crossover.shape[1]
                        != self.num_genes
                    ):
                        self.logger.error(
                            "Size mismatch between the crossover output {crossover_actual_size} and the expected "
                            "crossover output {crossover_expected_size}. It is expected that the offspring has "
                            "({num_genes}) genes but ({num_genes_produced}) produced.".format(
                                crossover_actual_size=self.last_generation_offspring_crossover.shape,
                                crossover_expected_size=(
                                    self.num_offspring,
                                    self.num_genes,
                                ),
                                num_genes=self.num_genes,
                                num_genes_produced=self.last_generation_offspring_crossover.shape[
                                    1
                                ],
                            )
                        )
                        raise ValueError(
                            "Size mismatch between the crossover output {crossover_actual_size} and the expected "
                            "crossover output {crossover_expected_size}. It is expected that the offspring has "
                            "({num_genes}) genes but ({num_genes_produced}) produced.".format(
                                crossover_actual_size=self.last_generation_offspring_crossover.shape,
                                crossover_expected_size=(
                                    self.num_offspring,
                                    self.num_genes,
                                ),
                                num_genes=self.num_genes,
                                num_genes_produced=self.last_generation_offspring_crossover.shape[
                                    1
                                ],
                            )
                        )

            # PyGAD 2.18.2 // The on_crossover() callback function is called even if crossover_type is None.
            if self.on_crossover is not None:
                self.on_crossover(self, self.last_generation_offspring_crossover)

            # If self.mutation_type=None, then no mutation is applied and thus no changes are applied to the offspring
            # created using the crossover operation. The offspring will be used unchanged in the next generation.
            if self.mutation_type is None:
                self.last_generation_offspring_mutation = (
                    self.last_generation_offspring_crossover
                )
            else:
                # Adding some variations to the offspring using mutation.
                if callable(self.mutation_type):
                    self.last_generation_offspring_mutation = self.mutation(
                        self.last_generation_offspring_crossover, self
                    )
                    if (
                        type(self.last_generation_offspring_mutation)
                        is not numpy.ndarray
                    ):
                        self.logger.error(
                            "The output of the mutation step is expected to be of type (numpy.ndarray) but "
                            "{last_generation_offspring_mutation_type} found.".format(
                                last_generation_offspring_mutation_type=type(
                                    self.last_generation_offspring_mutation
                                )
                            )
                        )
                        raise TypeError(
                            "The output of the mutation step is expected to be of type (numpy.ndarray) but "
                            "{last_generation_offspring_mutation_type} found.".format(
                                last_generation_offspring_mutation_type=type(
                                    self.last_generation_offspring_mutation
                                )
                            )
                        )
                else:
                    self.last_generation_offspring_mutation = self.mutation(
                        self.last_generation_offspring_crossover
                    )

                if self.last_generation_offspring_mutation.shape != (
                    self.num_offspring,
                    self.num_genes,
                ):
                    if (
                        self.last_generation_offspring_mutation.shape[0]
                        != self.num_offspring
                    ):
                        self.logger.error(
                            "Size mismatch between the mutation output {mutation_actual_size} and the expected mutation"
                            " output {mutation_expected_size}. It is expected to produce ({num_offspring}) offspring"
                            " but ({num_offspring_produced}) produced.".format(
                                mutation_actual_size=self.last_generation_offspring_mutation.shape,
                                mutation_expected_size=(
                                    self.num_offspring,
                                    self.num_genes,
                                ),
                                num_offspring=self.num_offspring,
                                num_offspring_produced=self.last_generation_offspring_mutation.shape[
                                    0
                                ],
                            )
                        )
                        raise ValueError(
                            "Size mismatch between the mutation output {mutation_actual_size} and the expected mutation"
                            " output {mutation_expected_size}. It is expected to produce ({num_offspring}) offspring"
                            " but ({num_offspring_produced}) produced.".format(
                                mutation_actual_size=self.last_generation_offspring_mutation.shape,
                                mutation_expected_size=(
                                    self.num_offspring,
                                    self.num_genes,
                                ),
                                num_offspring=self.num_offspring,
                                num_offspring_produced=self.last_generation_offspring_mutation.shape[
                                    0
                                ],
                            )
                        )
                    elif (
                        self.last_generation_offspring_mutation.shape[1]
                        != self.num_genes
                    ):
                        self.logger.error(
                            "Size mismatch between the mutation output {mutation_actual_size} and the expected mutation"
                            " output {mutation_expected_size}. It is expected that the offspring has ({num_genes})"
                            " genes but ({num_genes_produced}) produced.".format(
                                mutation_actual_size=self.last_generation_offspring_mutation.shape,
                                mutation_expected_size=(
                                    self.num_offspring,
                                    self.num_genes,
                                ),
                                num_genes=self.num_genes,
                                num_genes_produced=self.last_generation_offspring_mutation.shape[
                                    1
                                ],
                            )
                        )
                        raise ValueError(
                            "Size mismatch between the mutation output {mutation_actual_size} and the expected mutation"
                            " output {mutation_expected_size}. It is expected that the offspring has ({num_genes})"
                            " genes but ({num_genes_produced}) produced.".format(
                                mutation_actual_size=self.last_generation_offspring_mutation.shape,
                                mutation_expected_size=(
                                    self.num_offspring,
                                    self.num_genes,
                                ),
                                num_genes=self.num_genes,
                                num_genes_produced=self.last_generation_offspring_mutation.shape[
                                    1
                                ],
                            )
                        )

            # PyGAD 2.18.2 // The on_mutation() callback function is called even if mutation_type is None.
            if self.on_mutation is not None:
                self.on_mutation(self, self.last_generation_offspring_mutation)

            # Update the population attribute according to the offspring generated.
            if self.keep_elitism == 0:
                # If the keep_elitism parameter is 0, then the keep_parents parameter will be used to decide if the
                # parents are kept in the next generation.
                if self.keep_parents == 0:
                    self.population = self.last_generation_offspring_mutation
                elif self.keep_parents == -1:
                    # Creating the new population based on the parents and offspring.
                    self.population[
                        0 : self.last_generation_parents.shape[0], :
                    ] = self.last_generation_parents
                    self.population[
                        self.last_generation_parents.shape[0] :, :
                    ] = self.last_generation_offspring_mutation
                elif self.keep_parents > 0:
                    parents_to_keep, _ = self.steady_state_selection(
                        self.last_generation_fitness, num_parents=self.keep_parents
                    )
                    self.population[0 : parents_to_keep.shape[0], :] = parents_to_keep
                    self.population[
                        parents_to_keep.shape[0] :, :
                    ] = self.last_generation_offspring_mutation
            else:
                (
                    self.last_generation_elitism,
                    self.last_generation_elitism_indices,
                ) = self.steady_state_selection(
                    self.last_generation_fitness, num_parents=self.keep_elitism
                )
                self.population[
                    0 : self.last_generation_elitism.shape[0], :
                ] = self.last_generation_elitism
                self.population[
                    self.last_generation_elitism.shape[0] :, :
                ] = self.last_generation_offspring_mutation

            self.generations_completed = (
                generation + 1
            )  # The generations_completed attribute holds the number of the
            # last completed generation.

            self.previous_generation_fitness = self.last_generation_fitness.copy()
            # Measuring the fitness of each chromosome in the population. Save the fitness in the
            # last_generation_fitness attribute.

            # todo: ### Modified past this line:
            # since we are not interested in the fitness of the latest generation nor the solution just skip this
            # self.last_generation_fitness = self.cal_pop_fitness()
            # ### end of modification
            best_solution, best_solution_fitness, best_match_idx = self.best_solution(
                pop_fitness=self.last_generation_fitness
            )

            # Appending the best solution in the current generation to the best_solutions list.
            if self.save_best_solutions:
                self.best_solutions.append(best_solution)

            # If the on_generation attribute is not None, then cal the callback function after the generation.
            if self.on_generation is not None:
                r = self.on_generation(self)
                if isinstance(r, str) and r.lower() == "stop":
                    # Before aborting the loop, save the fitness value of the best solution.
                    # _, best_solution_fitness, _ = self.best_solution()
                    self.best_solutions_fitness.append(best_solution_fitness)
                    break

            if self.stop_criteria is not None:
                for criterion in self.stop_criteria:
                    if criterion[0] == "reach":
                        if max(self.last_generation_fitness) >= criterion[1]:
                            stop_run = True
                            break
                    elif criterion[0] == "saturate":
                        criterion[1] = int(criterion[1])
                        if self.generations_completed >= criterion[1]:
                            if (
                                self.best_solutions_fitness[
                                    self.generations_completed - criterion[1]
                                ]
                                - self.best_solutions_fitness[
                                    self.generations_completed - 1
                                ]
                            ) == 0:
                                stop_run = True
                                break

            if stop_run:
                break

            time.sleep(self.delay_after_gen)

        # Save the fitness of the last generation.
        if self.save_solutions:
            # self.solutions.extend(self.population.copy())
            population_as_list = self.population.copy()
            population_as_list = [list(item) for item in population_as_list]
            self.solutions.extend(population_as_list)

            self.solutions_fitness.extend(self.last_generation_fitness)

        # todo : if executing single_runs this will always append the solution a second time after every run
        # Save the fitness value of the best solution.
        # _, best_solution_fitness, _ = self.best_solution(pop_fitness=self.last_generation_fitness)
        # self.best_solutions_fitness.append(best_solution_fitness)
        # ### end of modification

        self.best_solution_generation = numpy.where(
            numpy.array(self.best_solutions_fitness)
            == numpy.max(numpy.array(self.best_solutions_fitness))
        )[0][0]
        # After the run() method completes, the run_completed flag is changed from False to True.
        self.run_completed = (
            True  # Set to True only after the run() method completes gracefully.
        )

        if self.on_stop is not None:
            self.on_stop(self, self.last_generation_fitness)

        # Converting the 'best_solutions' list into a NumPy array.
        self.best_solutions = numpy.array(self.best_solutions)

        # Converting the 'solutions' list into a NumPy array.
        # self.solutions = numpy.array(self.solutions)
