#include "naive_cuda_simulation.cuh"
#include "physics/gravitation.h"
#include "physics/mechanics.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_wrappers.cuh"


void NaiveCudaSimulation::allocate_device_memory(Universe& universe, void** d_weights, void** d_forces, void** d_velocities, void** d_positions){

}

void NaiveCudaSimulation::free_device_memory(void** d_weights, void** d_forces, void** d_velocities, void** d_positions){

}

void NaiveCudaSimulation::copy_data_to_device(Universe& universe, void* d_weights, void* d_forces, void* d_velocities, void* d_positions){
    //TO DO c)
    parprog_cudaMemcpy(d_weights, universe.weights.data(), universe.num_bodies * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double2> forces(universe.num_bodies);
    std::vector<double2> velocities(universe.num_bodies);
    std::vector<double2> positions(universe.num_bodies);

    for (int i = 0; i < universe.num_bodies; i++){
      forces[i] = {universe.forces[i][0], universe.forces[i][1]};
      velocities[i] = {universe.velocities[i][0], universe.velocities[i][1]};
      positions[i] = {universe.positions[i][0], universe.positions[i][1]};
    }

    parprog_cudaMemcpy(d_forces, forces.data(), universe.num_bodies * sizeof(double2), cudaMemcpyHostToDevice);
    parprog_cudaMemcpy(d_velocities, velocities.data(), universe.num_bodies * sizeof(double2), cudaMemcpyHostToDevice);
    parprog_cudaMemcpy(d_positions, positions.data(), universe.num_bodies * sizeof(double2), cudaMemcpyHostToDevice);
}

void NaiveCudaSimulation::copy_data_from_device(Universe& universe, void* d_weights, void* d_forces, void* d_velocities, void* d_positions){
    //To Do d)
    parprog_cudaMemcpy(universe.weights.data(), d_weights, universe.num_bodies * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double2> forces(universe.num_bodies);
    std::vector<double2> velocities(universe.num_bodies);
    std::vector<double2> positions(universe.num_bodies);

    parprog_cudaMemcpy(forces.data(), d_forces, universe.num_bodies * sizeof(double2), cudaMemcpyDeviceToHost);
    parprog_cudaMemcpy(velocities.data(), d_velocities, universe.num_bodies * sizeof(double2), cudaMemcpyDeviceToHost);
    parprog_cudaMemcpy(positions.data(), d_positions, universe.num_bodies * sizeof(double2), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < universe.num_bodies; ++i) {
        universe.forces[i] = {forces[i].x, forces[i].y};
        universe.velocities[i] = {velocities[i].x, velocities[i].y};
        universe.positions[i] = {positions[i].x, positions[i].y};
    }
}

__global__
void calculate_forces_kernel(std::uint32_t num_bodies, double2* d_positions, double* d_weights, double2* d_forces){

        std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= num_bodies) return;

        double2 pos_i = d_positions[i];
        double mass_i = d_weights[i];
        double2 force = {0.0, 0.0};

        for (std::uint32_t j = 0; j < num_bodies; j++) {
            if (i == j) continue;

            double2 pos_j = d_positions[j];
            double mass_j = d_weights[j];

            double dx = pos_j.x - pos_i.x;
            double dy = pos_j.y - pos_i.y;
            double dist_sq = dx * dx + dy * dy + 1e-9;
            double inv_dist = rsqrt(dist_sq);
            double f = G * mass_i * mass_j * inv_dist * inv_dist;

            force.x += f * dx * inv_dist;
            force.y += f * dy * inv_dist;
        }

        d_forces[i] = force;
    }

}

void NaiveCudaSimulation::calculate_forces(Universe& universe, void* d_positions, void* d_weights, void* d_forces){

    std::uint32_t num_bodies = universe.num_bodies;
    std::uint32_t threads_per_block = 256;
    std::uint32_t num_blocks = (num_bodies + threads_per_block - 1) / threads_per_block;

    calculate_forces_kernel<<<num_blocks, threads_per_block>>>(
        num_bodies, (double2*) d_positions, (double*) d_weights, (double2*) d_forces);
    cudaDeviceSynchronize();  // Warten auf Abschluss der Berechnung
}

__global__
void calculate_velocities_kernel(std::uint32_t num_bodies, double2* d_forces, double* d_weights, double2* d_velocities){
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bodies) return;

    double2 force = d_forces[i];
    double mass = d_weights[i];
    double2 velocity = d_velocities[i];

    double inv_mass = 1.0 / mass;
    velocity.x += force.x * inv_mass * epoch_in_seconds;
    velocity.y += force.y * inv_mass * epoch_in_seconds;

    d_velocities[i] = velocity;
}

void NaiveCudaSimulation::calculate_velocities(Universe& universe, void* d_forces, void* d_weights, void* d_velocities){
    std::uint32_t num_bodies = universe.num_bodies;
    std::uint32_t threads_per_block = 256;
    std::uint32_t num_blocks = (num_bodies + threads_per_block - 1) / threads_per_block;

    calculate_velocities_kernel<<<num_blocks, threads_per_block>>>(
        num_bodies, (double2*) d_forces, (double*) d_weights, (double2*) d_velocities);
    cudaDeviceSynchronize();
}

__global__
void calculate_positions_kernel(std::uint32_t num_bodies, double2* d_velocities, double2* d_positions){

        std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= num_bodies) return;

        double2 velocity = d_velocities[i];
        double2 position = d_positions[i];

        position.x += velocity.x * epoch_in_seconds;
        position.y += velocity.y * epoch_in_seconds;

        d_positions[i] = position;
}

void NaiveCudaSimulation::calculate_positions(Universe& universe, void* d_velocities, void* d_positions){

        std::uint32_t num_bodies = universe.num_bodies;
        std::uint32_t threads_per_block = 256;
        std::uint32_t num_blocks = (num_bodies + threads_per_block - 1) / threads_per_block;

        calculate_positions_kernel<<<num_blocks, threads_per_block>>>(
            num_bodies, (double2*) d_velocities, (double2*) d_positions);
        cudaDeviceSynchronize();
    }

}

void NaiveCudaSimulation::simulate_epochs(Plotter& plotter, Universe& universe, std::uint32_t num_epochs, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs){

}

__global__
void get_pixels_kernel(std::uint32_t num_bodies, double2* d_positions, std::uint8_t* d_pixels, std::uint32_t plot_width, std::uint32_t plot_height, double plot_bounding_box_x_min, double plot_bounding_box_x_max, double plot_bounding_box_y_min, double plot_bounding_box_y_max){

}

std::vector<std::uint8_t> NaiveCudaSimulation::get_pixels(std::uint32_t plot_width, std::uint32_t plot_height, BoundingBox plot_bounding_box, void* d_positions, std::uint32_t num_bodies){
    std::vector<std::uint8_t> pixels;
    return pixels;
}

__global__
void compress_pixels_kernel(std::uint32_t num_raw_pixels, std::uint8_t* d_raw_pixels, std::uint8_t* d_compressed_pixels){

}

void NaiveCudaSimulation::compress_pixels(std::vector<std::uint8_t>& raw_pixels, std::vector<std::uint8_t>& compressed_pixels){

}

void NaiveCudaSimulation::simulate_epoch(Plotter& plotter, Universe& universe, bool create_intermediate_plots, std::uint32_t plot_intermediate_epochs, void* d_weights, void* d_forces, void* d_velocities, void* d_positions){
    calculate_forces(universe, d_positions, d_weights, d_forces);
    calculate_velocities(universe, d_forces, d_weights, d_velocities);
    calculate_positions(universe, d_velocities, d_positions);

    universe.current_simulation_epoch++;
    if(create_intermediate_plots){
        if(universe.current_simulation_epoch % plot_intermediate_epochs == 0){
            std::vector<std::uint8_t> pixels = get_pixels(plotter.get_plot_width(), plotter.get_plot_height(), plotter.get_plot_bounding_box(), d_positions, universe.num_bodies);
            plotter.add_active_pixels_to_image(pixels);

            // This is a dummy to use compression in plotting, although not beneficial performance-wise
            // ----
            // std::vector<std::uint8_t> compressed_pixels;
            // compressed_pixels.resize(pixels.size()/8);
            // compress_pixels(pixels, compressed_pixels);
            // plotter.add_compressed_pixels_to_image(compressed_pixels);
            // ----

            plotter.write_and_clear();
        }
    }
}

void NaiveCudaSimulation::calculate_forces_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void* d_positions, void* d_weights, void* d_forces){
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_forces_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_positions, (double*) d_weights, (double2*) d_forces);
}

void NaiveCudaSimulation::calculate_velocities_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void* d_forces, void* d_weights, void* d_velocities){
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_velocities_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_forces, (double*) d_weights, (double2*) d_velocities);
}

void NaiveCudaSimulation::calculate_positions_kernel_test_adapter(std::uint32_t grid_dim, std::uint32_t block_dim, std::uint32_t num_bodies, void* d_velocities, void* d_positions){
    // adapter function used by automatic tests. DO NOT MODIFY.
    dim3 blockDim(block_dim);
    dim3 gridDim(grid_dim);
    calculate_positions_kernel<<<gridDim, blockDim>>>(num_bodies, (double2*) d_velocities, (double2*) d_positions);
}
