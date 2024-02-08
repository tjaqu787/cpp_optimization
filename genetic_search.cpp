#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <ctime>
#include <omp.h>

#include <sqlite3.h>
#include "pso.cpp"


// Use a standard library or a custom implementation for matrix operations
typedef std::vector<std::vector<int>> Matrix;

// Random number generator
std::default_random_engine generator;
std::uniform_int_distribution<int> distribution(0, 10);

// Function to initialize population
Matrix initialize_population(int pop_size, int num_vars = 15) {
    Matrix population(pop_size, std::vector<int>(num_vars));
    for (auto& row : population)
        for (auto& val : row)
            val = distribution(generator);
    return population;
}

std::pair<std::vector<int>, std::vector<std::vector<int64_t>>> function_to_optimize(
    auto params,
    int iterations = 2000,
    int num_particles = 10000,
    double inertia = 0.5,
    double cognitive = 0.5,
    double social = 1.5){

    std::vector<int> score_result(params[0].size());
    std::vector<std::vector<int64_t>> position_result(params[0].size());

    #pragma omp parallel for
    for (int i = 0; i < params.size(); ++i) {
            auto result = PSO(params[i], iterations, params[1].size() +1, num_particles, inertia, cognitive, social);                  
            score_result[i] = result.first;
            position_result[i] = result.second;}
    return std::pair(score_result,position_result);
}

void send_to_db(double cost, const std::vector<int>& func_params,std::vector<int64_t>& best_position) {
    
    if (cost <= 8500){
    // Create a timestamp
    std::time_t timestamp = std::time(nullptr);
    // Prepare the SQL statement
    const char* sql = "INSERT INTO lottmax_accuracy(timestamp, cost, n_gen, params) VALUES (?, ?, ?, ?)";
    sqlite3_stmt* stmt;
    sqlite3* db;

    int result = sqlite3_open("../accuracy.db", &db);
        // SQL statement to create the table if it doesn't exist
    const char* createTableSQL = R"(
        CREATE TABLE IF NOT EXISTS lottmax_accuracy(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            cost REAL NOT NULL,
            n_gen INTEGER NOT NULL,
            params TEXT NOT NULL
        )
    )";
    char* errMsg = nullptr;

    // Execute the SQL statement to create the table if needed
    result = sqlite3_exec(db, createTableSQL, nullptr, nullptr, &errMsg);
    if (result != SQLITE_OK) {
        std::cerr << "SQL error: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        sqlite3_close(db);}

    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) {
        std::cerr << "Cannot prepare statement: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_finalize(stmt);}

    // Bind values to the placeholders in the SQL statement
    sqlite3_bind_int64(stmt, 1, timestamp);
    sqlite3_bind_double(stmt, 2, cost);
    sqlite3_bind_double(stmt, 3, best_position.back()); 
    best_position.pop_back();
    
    // Convert parameters to a string
    std::string params_str; 
    for (auto param : func_params) {
        params_str += std::to_string(param) + ",";
    }    
    for (auto param : best_position) {
        params_str += std::to_string(param) + ",";
    } 

    sqlite3_bind_text(stmt, 4, params_str.c_str(), -1, SQLITE_STATIC);

    // Execute the prepared statement
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        std::cerr << "Execution failed: " << sqlite3_errmsg(db) << std::endl;
    }

    // Finalize the statement
    sqlite3_finalize(stmt);
    sqlite3_close(db);
}}

// Function to select parents
Matrix select_parents(const Matrix& population, const std::vector<int>& fitnesses, auto& best_position, int num_parents) {

    std::vector<int> indices(population.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on fitness in descending order
    std::sort(indices.begin(), indices.end(), [&](int i, int j) { return fitnesses[i] > fitnesses[j]; });

    Matrix parents;
    for (int i = 0; i < num_parents; ++i) { 
        int index = indices[i];
        parents.push_back(population[index]);
        
        int fitness = fitnesses[index];
        auto position = best_position[index];
        std::vector<int> parameters = population[index]; 
        send_to_db(fitness, parameters,position);
    }

    return parents;}

// Function for crossover
std::pair<std::vector<int>, std::vector<int>> crossover(const std::vector<int>& parent1, const std::vector<int>& parent2) {
    int crossover_point = distribution(generator) % (parent1.size() - 2) + 1;
    std::vector<int> child1(parent1.begin(), parent1.begin() + crossover_point);
    child1.insert(child1.end(), parent2.begin() + crossover_point, parent2.end());

    std::vector<int> child2(parent2.begin(), parent2.begin() + crossover_point);
    child2.insert(child2.end(), parent1.begin() + crossover_point, parent1.end());

    return {child1, child2};
}

// Function for mutation
std::vector<int> mutate(const std::vector<int>& individual, double mutation_rate = 0.1) {
    std::vector<int> mutated(individual);
    for (auto& val : mutated) {
        if ((double)rand() / RAND_MAX < mutation_rate)
            val = distribution(generator);
    }
    return mutated;
}

// Genetic algorithm function
void genetic_algorithm(
    int pop_size = 100000, 
    int num_generations = 100000,
    int num_parents = 100,
    double mutation_rate = 0.3) {

        
    Matrix population = initialize_population(pop_size);

    for (int generation = 0; generation < num_generations; ++generation) {
        auto [fitnesses,positions] = function_to_optimize(population);

        Matrix parents = select_parents(population, fitnesses,positions, num_parents);

        Matrix offspring;
        while (offspring.size() < pop_size - num_parents) {
            int p1 = distribution(generator) % num_parents;
            int p2 = distribution(generator) % num_parents;
            auto [child1, child2] = crossover(parents[p1], parents[p2]);
            offspring.push_back(mutate(child1, mutation_rate));
            offspring.push_back(mutate(child2, mutation_rate));
        }

        // Update the population
        population = parents;
        population.insert(population.end(), offspring.begin(), offspring.begin() + (pop_size - num_parents));
    }
}


