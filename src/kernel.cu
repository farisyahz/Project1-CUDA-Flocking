#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#include <glm/glm.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_pos2;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_pos2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos2 failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // Initialize velocity to 0
  cudaMemset(dev_vel1, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel1 failed!");

  cudaMemset(dev_vel2, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum = glm::vec3(-halfGridWidth);

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**) &dev_particleArrayIndices, N * sizeof(int)); // boids i points to the index of the val & pos -> we have to initialize with 0, 1, 2, 3, etc
  cudaMalloc((void**) &dev_particleGridIndices, N * sizeof(int)); // boids i is located in which grid cell
  cudaMalloc((void**) &dev_gridCellStartIndices, gridCellCount * sizeof(int)); // cell i has boid idx start where
  cudaMalloc((void**) &dev_gridCellEndIndices, gridCellCount * sizeof(int)); // cell i has boid idx ends where

  // cudaMalloc((void**) &dev_thrust_particleArrayIndices, N * sizeof(thrust::device_ptr<int>));
  // cudaMalloc((void**) &dev_thrust_particleGridIndices, N * sizeof(thrust::device_ptr<int>));

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

__device__ glm::vec3 rule1(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel);
__device__ glm::vec3 rule2(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel);
__device__ glm::vec3 rule3(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel);

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {

  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  glm::vec3 v1 = rule1(N, iSelf, pos, vel);
  // Rule 2: boids try to stay a distance d away from each other
  glm::vec3 v2 = rule2(N, iSelf, pos, vel);
  // Rule 3: boids try to match the speed of surrounding boids
  glm::vec3 v3 = rule3(N, iSelf, pos, vel);

  return (v1 + v2 + v3);
}

// RULE 1 [NAIVE]
__device__ glm::vec3 rule1(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel){
  glm::vec3 percieved_center(0.0f);
  size_t number_of_neighbors = 0;

  for (int i = 0; i < N; ++i){
    if (iSelf != i && glm::distance(pos[i], pos[iSelf]) < rule1Distance){
      percieved_center += pos[i];
      ++number_of_neighbors;
    }
  }

  if (number_of_neighbors == 0) return glm::vec3(0.0f);

  percieved_center /= static_cast<float>(number_of_neighbors);
  return (percieved_center - pos[iSelf]) * rule1Scale;
}

// RULE 2 [NAIVE]
__device__ glm::vec3 rule2(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel){
  glm::vec3 c(0.0f);

  for (int i = 0; i < N; ++i){
    if (iSelf != i && glm::distance(pos[i], pos[iSelf]) < rule2Distance){
      c -= (pos[i] - pos[iSelf]);
    }
  }

  return c * rule2Scale;
}

// RULE 3 [NAIVE]
__device__ glm::vec3 rule3(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel){
  glm::vec3 percieved_vel(0.0f);
  size_t number_of_neighbors = 0;

  for (int i = 0; i < N; ++i){
    if (iSelf != i && glm::distance(pos[i], pos[iSelf]) < rule3Distance){
      percieved_vel += vel[i];
      ++number_of_neighbors;
    }
  }

  if (number_of_neighbors == 0) return glm::vec3(0.0f);

  percieved_vel /= static_cast<float>(number_of_neighbors);
  return percieved_vel * rule3Scale;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= N) return;
  // The rules return a velocity change, not the complete new velocity.
  glm::vec3 v_res = vel1[index] + computeVelocityChange(N, index, pos, vel1);
  float speed = glm::length(v_res);
  if (speed > maxSpeed) {
    v_res = v_res / speed * maxSpeed;
  }

  // Record the new velocity into vel2. Question: why NOT vel1? because vel1 would still being read by other thread, 
  // so we change it per batch, which is called ping pong buffers
  vel2[index] = v_res;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= N) return;

    glm::vec3 gridPosition = (pos[index] - gridMin) * inverseCellWidth;

    int x = static_cast<int>(gridPosition.x);
    int y = static_cast<int>(gridPosition.y);
    int z = static_cast<int>(gridPosition.z);

    indices[index] = index;
    gridIndices[index] = gridIndex3Dto1D(x, y, z, gridResolution);
  }  

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= N) {
    return;
  }

  int cell = particleGridIndices[index];

  if (index == 0)
    gridCellStartIndices[cell] = 0;

  if (index < N - 1) {
    int nextCell = particleGridIndices[index + 1];

    if (cell != nextCell) {
      gridCellEndIndices[cell] = index + 1;
      gridCellStartIndices[nextCell] = index + 1;
    }
  }

  if (index == N - 1)
    gridCellEndIndices[cell] = N;
}

__global__ void kernUpdateVelNeighborSearchScattered(
    int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int *gridCellStartIndices, int *gridCellEndIndices,
    int *particleArrayIndices,
    glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {

  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= N) {
    return;
  }

  const glm::vec3 selfPos = pos[index];

  // The neighborhood radius stays fixed when experimenting with cell width.
  const float searchRadius = fmaxf(rule1Distance, fmaxf(rule2Distance, rule3Distance));

  // Find the range of grid cells touched by this boid's neighborhood.
  int minX = static_cast<int>(
      (selfPos.x - searchRadius - gridMin.x) * inverseCellWidth);
  int minY = static_cast<int>(
      (selfPos.y - searchRadius - gridMin.y) * inverseCellWidth);
  int minZ = static_cast<int>(
      (selfPos.z - searchRadius - gridMin.z) * inverseCellWidth);

  int maxX = static_cast<int>(
      (selfPos.x + searchRadius - gridMin.x) * inverseCellWidth);
  int maxY = static_cast<int>(
      (selfPos.y + searchRadius - gridMin.y) * inverseCellWidth);
  int maxZ = static_cast<int>(
      (selfPos.z + searchRadius - gridMin.z) * inverseCellWidth);

  // Keep cell coordinates inside the grid.
  minX = imax(0, minX);
  minY = imax(0, minY);
  minZ = imax(0, minZ);

  maxX = imin(gridResolution - 1, maxX);
  maxY = imin(gridResolution - 1, maxY);
  maxZ = imin(gridResolution - 1, maxZ);

  // Accumulators for the three boids rules.
  glm::vec3 perceivedCenter(0.0f);
  glm::vec3 separation(0.0f);
  glm::vec3 perceivedVelocity(0.0f);

  int rule1NeighborCount = 0;
  int rule3NeighborCount = 0;

  // x is innermost because gridIndex3Dto1D makes adjacent x cells
  // contiguous in memory.
  for (int z = minZ; z <= maxZ; ++z) {
    for (int y = minY; y <= maxY; ++y) {
      for (int x = minX; x <= maxX; ++x) {
        int cellIndex =
            gridIndex3Dto1D(x, y, z, gridResolution);

        int start = gridCellStartIndices[cellIndex];
        int end = gridCellEndIndices[cellIndex];

        // Empty cells should have start/end set to -1.
        if (start == -1 || end == -1) {
          continue;
        }

        // These are sorted-array indices. Look up the actual boid index
        // through particleArrayIndices.
        for (int sortedIndex = start;
             sortedIndex < end;
             ++sortedIndex) {

          int otherIndex = particleArrayIndices[sortedIndex];

          if (otherIndex == index) {
            continue;
          }

          glm::vec3 offset = pos[otherIndex] - selfPos;
          float distance = glm::length(offset);

          // Rule 1
          if (distance < rule1Distance) {
            perceivedCenter += pos[otherIndex];
            ++rule1NeighborCount;
          }

          // Rule 2
          if (distance < rule2Distance) {
            separation -= offset;
          }

          // Rule 3
          if (distance < rule3Distance) {
            perceivedVelocity += vel1[otherIndex];
            ++rule3NeighborCount;
          }
        }
      }
    }
  }

  glm::vec3 velocityChange(0.0f);

  // Avoid division by zero when a boid has no neighbors.
  if (rule1NeighborCount > 0) {
    perceivedCenter /= static_cast<float>(rule1NeighborCount);
    velocityChange +=
        (perceivedCenter - selfPos) * rule1Scale;
  }

  velocityChange += separation * rule2Scale;

  if (rule3NeighborCount > 0) {
    perceivedVelocity /= static_cast<float>(rule3NeighborCount);
    velocityChange += perceivedVelocity * rule3Scale;
  }

  // Apply the rule results to the old velocity.
  glm::vec3 newVelocity = vel1[index] + velocityChange;

  // Clamp the final speed.
  float speed = glm::length(newVelocity);
  if (speed > maxSpeed) {
    newVelocity = newVelocity / speed * maxSpeed;
  }

  vel2[index] = newVelocity;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= N) return;

  glm::vec3 selfPos = pos[index];
  float searchRadius = fmaxf(rule1Distance, fmaxf(rule2Distance, rule3Distance));

  int minX = static_cast<int>((selfPos.x - searchRadius - gridMin.x) * inverseCellWidth);
  int minY = static_cast<int>((selfPos.y - searchRadius - gridMin.y) * inverseCellWidth);
  int minZ = static_cast<int>((selfPos.z - searchRadius - gridMin.z) * inverseCellWidth);
  int maxX = static_cast<int>((selfPos.x + searchRadius - gridMin.x) * inverseCellWidth);
  int maxY = static_cast<int>((selfPos.y + searchRadius - gridMin.y) * inverseCellWidth);
  int maxZ = static_cast<int>((selfPos.z + searchRadius - gridMin.z) * inverseCellWidth);

  minX = imax(0, minX);
  minY = imax(0, minY);
  minZ = imax(0, minZ);
  maxX = imin(gridResolution - 1, maxX);
  maxY = imin(gridResolution - 1, maxY);
  maxZ = imin(gridResolution - 1, maxZ);

  glm::vec3 perceivedCenter(0.0f);
  glm::vec3 separation(0.0f);
  glm::vec3 perceivedVelocity(0.0f);
  int rule1NeighborCount = 0;
  int rule3NeighborCount = 0;

  for (int z = minZ; z <= maxZ; ++z) {
    for (int y = minY; y <= maxY; ++y) {
      for (int x = minX; x <= maxX; ++x) {
        int cellIndex = gridIndex3Dto1D(x, y, z, gridResolution);
        int start = gridCellStartIndices[cellIndex];
        int end = gridCellEndIndices[cellIndex];

        if (start == -1 || end == -1) continue;

        for (int otherIndex = start; otherIndex < end; ++otherIndex) {
          if (otherIndex == index) continue;

          glm::vec3 offset = pos[otherIndex] - selfPos;
          float distance = glm::length(offset);

          if (distance < rule1Distance) {
            perceivedCenter += pos[otherIndex];
            ++rule1NeighborCount;
          }

          if (distance < rule2Distance)
            separation -= offset;

          if (distance < rule3Distance) {
            perceivedVelocity += vel1[otherIndex];
            ++rule3NeighborCount;
          }
        }
      }
    }
  }

  glm::vec3 velocityChange(0.0f);

  if (rule1NeighborCount > 0) {
    perceivedCenter /= static_cast<float>(rule1NeighborCount);
    velocityChange += (perceivedCenter - selfPos) * rule1Scale;
  }

  velocityChange += separation * rule2Scale;

  if (rule3NeighborCount > 0) {
    perceivedVelocity /= static_cast<float>(rule3NeighborCount);
    velocityChange += perceivedVelocity * rule3Scale;
  }

  glm::vec3 newVelocity = vel1[index] + velocityChange;
  float speed = glm::length(newVelocity);

  if (speed > maxSpeed)
    newVelocity = newVelocity / speed * maxSpeed;

  vel2[index] = newVelocity;
}

/**
 * Shared-memory coherent-grid neighbor search.
 *
 * One block owns a grid cell. The block cooperatively loads neighboring boids
 * into shared-memory tiles, so every boid in the owned cell reuses the same
 * position and velocity reads. A full 3x3x3 neighborhood is required because
 * boids in the owned cell can sit on different sides of its center.
 */
__global__ void kernUpdateVelNeighborSearchCoherentShared(
    int gridResolution, int cellCount,
    const int *gridCellStartIndices, const int *gridCellEndIndices,
    const glm::vec3 *pos, const glm::vec3 *vel1, glm::vec3 *vel2) {

  extern __shared__ float sharedStorage[];
  float3 *sharedPos = reinterpret_cast<float3 *>(sharedStorage);
  float3 *sharedVel = sharedPos + blockDim.x;

  // A grid-stride loop avoids launching tens of thousands of mostly empty
  // blocks when the uniform grid is sparse.
  for (int cell = blockIdx.x; cell < cellCount; cell += gridDim.x) {
    const int ownStart = gridCellStartIndices[cell];
    const int ownEnd = gridCellEndIndices[cell];
    if (ownStart < 0 || ownEnd < 0) continue;

    const int cellX = cell % gridResolution;
    const int cellY = (cell / gridResolution) % gridResolution;
    const int cellZ = cell / (gridResolution * gridResolution);

    // Process cells larger than one CUDA block in consecutive batches. Every
    // thread participates in the barriers, including inactive lanes.
    for (int batchStart = ownStart; batchStart < ownEnd;
         batchStart += blockDim.x) {
      const int self = batchStart + threadIdx.x;
      const bool active = self < ownEnd;
      const glm::vec3 selfPos = active ? pos[self] : glm::vec3(0.0f);

      glm::vec3 perceivedCenter(0.0f);
      glm::vec3 separation(0.0f);
      glm::vec3 perceivedVelocity(0.0f);
      int rule1NeighborCount = 0;
      int rule3NeighborCount = 0;

      for (int z = imax(0, cellZ - 1);
           z <= imin(gridResolution - 1, cellZ + 1); ++z) {
        for (int y = imax(0, cellY - 1);
             y <= imin(gridResolution - 1, cellY + 1); ++y) {
          for (int x = imax(0, cellX - 1);
               x <= imin(gridResolution - 1, cellX + 1); ++x) {
            const int neighborCell =
                gridIndex3Dto1D(x, y, z, gridResolution);
            const int start = gridCellStartIndices[neighborCell];
            const int end = gridCellEndIndices[neighborCell];
            if (start < 0 || end < 0) continue;

            for (int tileStart = start; tileStart < end;
                 tileStart += blockDim.x) {
              const int tileLength =
                  imin(static_cast<int>(blockDim.x), end - tileStart);

              if (threadIdx.x < tileLength) {
                const glm::vec3 p = pos[tileStart + threadIdx.x];
                const glm::vec3 v = vel1[tileStart + threadIdx.x];
                sharedPos[threadIdx.x] = make_float3(p.x, p.y, p.z);
                sharedVel[threadIdx.x] = make_float3(v.x, v.y, v.z);
              }
              __syncthreads();

              if (active) {
                for (int j = 0; j < tileLength; ++j) {
                  if (tileStart + j == self) continue;

                  const glm::vec3 otherPos(
                      sharedPos[j].x, sharedPos[j].y, sharedPos[j].z);
                  const glm::vec3 otherVel(
                      sharedVel[j].x, sharedVel[j].y, sharedVel[j].z);
                  const glm::vec3 offset = otherPos - selfPos;
                  const float distance = glm::length(offset);

                  if (distance < rule1Distance) {
                    perceivedCenter += otherPos;
                    ++rule1NeighborCount;
                  }
                  if (distance < rule2Distance) separation -= offset;
                  if (distance < rule3Distance) {
                    perceivedVelocity += otherVel;
                    ++rule3NeighborCount;
                  }
                }
              }
              __syncthreads();
            }
          }
        }
      }

      if (active) {
        glm::vec3 velocityChange(0.0f);
        if (rule1NeighborCount > 0) {
          perceivedCenter /= static_cast<float>(rule1NeighborCount);
          velocityChange +=
              (perceivedCenter - selfPos) * rule1Scale;
        }
        velocityChange += separation * rule2Scale;
        if (rule3NeighborCount > 0) {
          perceivedVelocity /= static_cast<float>(rule3NeighborCount);
          velocityChange += perceivedVelocity * rule3Scale;
        }

        glm::vec3 newVelocity = vel1[self] + velocityChange;
        const float speed = glm::length(newVelocity);
        if (speed > maxSpeed)
          newVelocity = newVelocity / speed * maxSpeed;
        vel2[self] = newVelocity;
      }
    }
  }
}

__global__ void kernShufflePosVel(
  int N, int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel,
  glm::vec3 *shuffledPos, glm::vec3 *shuffledVel) {

  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= N) return;

  int sourceIndex = particleArrayIndices[index];
  shuffledPos[index] = pos[sourceIndex];
  shuffledVel[index] = vel[sourceIndex];
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  dim3 fullBlockPerGrid((numObjects + blockSize - 1) / blockSize);

  kernUpdateVelocityBruteForce<<<fullBlockPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);

  kernUpdatePos<<<fullBlockPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2); 
  // TODO-1.2 ping-pong the velocity buffers
  glm::vec3* temp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = temp;
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

  dim3 fullBlockPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 fullCellBlocksPerGrid((gridCellCount + blockSize - 1) / blockSize);

  kernComputeIndices<<<fullBlockPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  kernResetIntBuffer<<<fullCellBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<fullCellBlocksPerGrid, blockSize>>>(gridCellCount, dev_gridCellEndIndices, -1);

  kernIdentifyCellStartEnd<<<fullBlockPerGrid, blockSize>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

  kernUpdateVelNeighborSearchScattered<<<fullBlockPerGrid,blockSize>>>(
    numObjects,
    gridSideCount,
    gridMinimum,
    gridInverseCellWidth,
    gridCellWidth,
    dev_gridCellStartIndices,
    dev_gridCellEndIndices,
    dev_particleArrayIndices,
    dev_pos,
    dev_vel1,
    dev_vel2
  );

  kernUpdatePos<<<fullBlockPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);

  glm::vec3* temp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = temp;
}

static void stepSimulationCoherentGridImpl(float dt, bool useSharedMemory) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.

  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 fullCellBlocksPerGrid((gridCellCount + blockSize - 1) / blockSize);

  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
    dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  thrust::sort_by_key(
    dev_thrust_particleGridIndices,
    dev_thrust_particleGridIndices + numObjects,
    dev_thrust_particleArrayIndices);

  kernResetIntBuffer<<<fullCellBlocksPerGrid, blockSize>>>(
    gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<fullCellBlocksPerGrid, blockSize>>>(
    gridCellCount, dev_gridCellEndIndices, -1);

  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, dev_particleGridIndices,
    dev_gridCellStartIndices, dev_gridCellEndIndices);

  kernShufflePosVel<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, dev_particleArrayIndices,
    dev_pos, dev_vel1, dev_pos2, dev_vel2);

  if (useSharedMemory) {
    const int cellBlocks = imin(
        gridCellCount, imax(1, 2 * static_cast<int>(fullBlocksPerGrid.x)));
    const size_t sharedBytes =
        2 * blockSize * sizeof(float3);
    kernUpdateVelNeighborSearchCoherentShared
        <<<cellBlocks, blockSize, sharedBytes>>>(
            gridSideCount, gridCellCount,
            dev_gridCellStartIndices, dev_gridCellEndIndices,
            dev_pos2, dev_vel2, dev_vel1);
  } else {
    kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, blockSize>>>(
      numObjects,
      gridSideCount,
      gridMinimum,
      gridInverseCellWidth,
      gridCellWidth,
      dev_gridCellStartIndices,
      dev_gridCellEndIndices,
      dev_pos2,
      dev_vel2,
      dev_vel1);
  }

  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(
    numObjects, dt, dev_pos2, dev_vel1);

  glm::vec3 *temp = dev_pos;
  dev_pos = dev_pos2;
  dev_pos2 = temp;

  checkCUDAErrorWithLine("stepSimulationCoherentGrid failed!");
}

void Boids::stepSimulationCoherentGrid(float dt) {
  stepSimulationCoherentGridImpl(dt, false);
}

void Boids::stepSimulationCoherentGridShared(float dt) {
  stepSimulationCoherentGridImpl(dt, true);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);
  cudaFree(dev_pos2);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
