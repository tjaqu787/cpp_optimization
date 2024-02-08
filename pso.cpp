#include <eigen3/Eigen/Dense>
#include <iostream>
#include <random>
#include <limits>
#include <vector>

///typedef Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXu32;
typedef std::vector<std::vector<int64_t>> MatrixXu32;

// Initialize particles
void initialize_particles(auto& positions, auto& velocities, uint32_t lower_bound, uint32_t upper_bound) {
    //MatrixXu32 positions(dimensions, num_particles);
    std::random_device rd;
    std::mt19937 gen(rd());

    // to impliment a vector of bounds put this in the loop then iterate 
    // through the elements of the vector of bounds
    std::uniform_int_distribution<uint32_t> dis(lower_bound, upper_bound);

    auto dimensions = positions[0].size();
    auto num_particles = positions.size() ;
    for (size_t i = 0; i < num_particles; ++i) { 

        for (size_t j = 0; j < dimensions; ++j){ 
            positions[i][j] = dis(gen);}}}
            
// Update particles
void update_particles(MatrixXu32& positions,auto& velocities, const auto& best_position, double inertia, double cognitive, double social) {

    auto num_particles = positions.size();
    auto dimensions = positions[0].size();

    
    static std::mt19937 gen(std::time(nullptr));
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (size_t i = 0; i < num_particles; ++i) { 
        for (size_t j = 0; j < dimensions; ++j){ 
            // Update velocity
            velocities[i][j] = static_cast<int32_t>(inertia * velocities[i][j] +
                cognitive * dis(gen) * (best_position[j] - positions[i][j]) +
                social * dis(gen) * (best_position[j] - positions[i][j]));

            // Update position
            positions[i][j] += velocities[i][j];

            // add proper boundry condition 
            if (positions[i][j] ==0){
                positions[i][j]=positions[i][j]+1;
            }
        }            

    }

}



// PSO function
std::pair<int, std::vector<int64_t>> PSO(auto function_to_optimize,
    auto func_params, int iters, int dimensions, 
    int num_particles, double inertia, double cognitive, double social) {

    double best_score = std::numeric_limits<double>::lowest();
    std::vector<int> score_matrix;
    score_matrix.reserve(num_particles);

    MatrixXu32 positions(num_particles, std::vector<int64_t>(dimensions));
    std::vector<std::vector<double>> velocities(num_particles, std::vector<double>(dimensions));


    uint32_t lower_bound = 1;
    uint32_t upper_bound = std::numeric_limits<uint32_t>::max();
                   
    //MatrixXu32 best_position;
    std::vector<int64_t> best_position;
     initialize_particles(positions,velocities, lower_bound, upper_bound);

    for (int i = 0; i < iters; ++i) {

        double current_score;

        score_matrix = function_to_optimize(func_params,positions,target);

        // Find the current maximum score in the score_matrix
        current_score = *std::max_element(score_matrix.begin(), score_matrix.end());
        if (current_score > best_score) {
            best_score = current_score;
            int best_position_idx = std::distance(score_matrix.begin(), std::max_element(score_matrix.begin(), score_matrix.end()));

            // Copy the corresponding position from the positions matrix
            best_position = positions[best_position_idx];

        
        }

        update_particles(positions,velocities, best_position, inertia, cognitive, social);
    }

    return std::pair(best_score,best_position);
}
