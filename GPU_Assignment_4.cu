// %%writefile dhanush.cu
#include <bits/stdc++.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define MAX_PATH_LENGTH 1000
#define MAX_SHELTER_LIMIT 1000
#define MAX_CITY_COUNT 1000

using namespace std;

static const long long INF = (long long)4e18;
// Structure for an edge in the graph
struct Edge {
    int targetNode;
    int weight;
};

// Structure for evacuation result

struct EvacuationResult {
    int path_size;
    int drops_size;
    int path[MAX_PATH_LENGTH];
    long long drops[MAX_PATH_LENGTH][3];
};


// CUDA kernel for evacuation
__global__ void kernel1(int totalCities, int totalShelters, int totalPopulated,
    int *d_populated_city, int *d_popcity_primepop, int *d_popcity_elderpop, 
    int *d_shelter_city, int *d_shelter_capacity, int *d_cityToShelterIndex, 
    int *d_edges_target, int *d_edges_len, int *d_edge_rowstart, int *d_edge_count, 
    EvacuationResult *d_results, int elderlyDistanceLimit)
 {  
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= totalPopulated) return;

    // Prepare result container for this thread
    EvacuationResult &res = d_results[id];
    res.drops_size = 0;
    res.path_size = 0;

    // Initialize evacuation state for this thread
    long long remaining_elderly = d_popcity_elderpop[id];
    long long remaining_prime = d_popcity_primepop[id];
    int source_city = d_populated_city[id];
    int current_city = source_city;

    // Begin path with source city
    res.path[res.path_size++] = current_city;
    
    // Track used shelters
    int no_of_shelters_used = 0;
    bool shelter_used[MAX_SHELTER_LIMIT] = {false}; // Assuming max shelters is less than 1000
    
    // Track the first non-shelter city encountered during the path
    int first_non_shelter_city = -1;
    bool is_source_city_shelter = false;

    // Check if source city is a shelter
    for (int i = 0; i < totalShelters; i++) {
        if (source_city == d_shelter_city[i]) {
            is_source_city_shelter = true;
            break;
        }
    }
    
    // If source is not a shelter, it's our first non-shelter city
    if (!is_source_city_shelter) {
        first_non_shelter_city = source_city;
    }

    while (remaining_prime > 0 || remaining_elderly > 0)
    {   
        bool visited_flag[MAX_CITY_COUNT] = {false};
        long long min_distance[MAX_CITY_COUNT];
        int previous_city[MAX_CITY_COUNT];
        int selected_index = -1;
        int selected_city = -1;

        // Initialize distances and predecessors
        for (int i = 0; i < totalCities; i++) {
            min_distance[i] = (i == current_city) ? 0 : INF;
            previous_city[i] = -1;
        } 
        
        for (int i = 0; i < totalCities; i++) {
            // Select the unvisited node with the smallest distance
            long long smallest_dist = INF;
            int nearest_city = -1;
            for (int j = 0; j < totalCities; j++) {
                if (!visited_flag[j] && min_distance[j] < smallest_dist) {
                    nearest_city = j;
                    smallest_dist = min_distance[j];
                }
            }
    
            if (smallest_dist == INF) break;
            if (nearest_city == -1) break;
        
            // Explore neighbors of the selected city
            int edge_start_idx = d_edge_rowstart[nearest_city];
            int edge_end_idx = edge_start_idx + d_edge_count[nearest_city];
        
            int edge_idx = edge_start_idx;
            while (edge_idx < edge_end_idx) {
                int road_len = d_edges_len[edge_idx];
                int neighbor = d_edges_target[edge_idx];

                int new_distance = min_distance[nearest_city] + road_len;
                if (new_distance < min_distance[neighbor]) {
                    previous_city[neighbor] = nearest_city;
                    min_distance[neighbor] = new_distance;
                }
                edge_idx++;
            }
            visited_flag[nearest_city] = true;
        }
        
        // Select nearest available shelter not yet used
        long long nearest_distance = INF;

        int shelter_idx = 0;
        while (shelter_idx < totalShelters) {
            if (shelter_used[shelter_idx] == false) {
                int shelter_city = d_shelter_city[shelter_idx];
                int available_capacity = atomicAdd(&d_shelter_capacity[shelter_idx], 0); // Read-only

                if (available_capacity > 0 && min_distance[shelter_city] < nearest_distance) {
                    selected_city = shelter_city;
                    selected_index = shelter_idx;
                    nearest_distance = min_distance[shelter_city];
                }
            }
            shelter_idx++;
        }
        
        if (selected_index >= 0) 
        {
            shelter_used[selected_index] = true;
            no_of_shelters_used++;
        } 
        else
        {
            // No reachable shelter found
            break;
        }        
        
        int segment_length = 0;
        int path_segment[MAX_CITY_COUNT];

        // Trace path backward from selected city to current location
        for (int v = selected_city; v != -1; v = previous_city[v]) {
            path_segment[segment_length] = v;
            segment_length++;
            if (v == current_city) break;
        }
        bool last_city_check=(path_segment[segment_length - 1] != current_city);
        if (last_city_check) break;

        int mid = segment_length / 2;
        for (int i = mid - 1; i >= 0; i--) {
            int opposite = segment_length - 1 - i;
            int temp = path_segment[i];
            path_segment[i] = path_segment[opposite];
            path_segment[opposite] = temp;
        }
        
        // Check path segment for non-shelter cities and record the first one found
        if (first_non_shelter_city == -1) {
            for (int i = 0; i < segment_length; i++) {
                int city = path_segment[i];
                bool is_shelter = false;
                
                for (int s = 0; s < totalShelters; s++) {
                    if (city == d_shelter_city[s]) {
                        is_shelter = true;
                        break;
                    }
                }
                
                if (!is_shelter) {
                    first_non_shelter_city = city;
                    break;
                }
            }
        }

        // Append reconstructed segment to result path, skipping the first node
        for (int idx = 0; idx < segment_length; idx++) {
            if (idx == 0) continue;
            res.path[res.path_size++] = path_segment[idx];
        }

        // Check if elderly can reach the selected shelter
        int farthest_reachable = path_segment[0];
        long long accumulated_distance = 0;

        // Find last non-shelter city before distance limit
        int last_non_shelter_before_limit = -1;

        for (int i = 0; i + 1 < segment_length; i++) {
            int weight = -1;
            int from = path_segment[i];
            int to = path_segment[i + 1];

            // Search for edge weight between 'from' and 'to'
            int start = d_edge_rowstart[from];
            int end = start + d_edge_count[from];
            int e = start;

            while (e < end) {
                if (d_edges_target[e] != to) {
                    ++e;
                    continue;
                }

                weight = d_edges_len[e];
                accumulated_distance += weight;
                break;
            }

            // Before updating farthest_reachable, check if current city is non-shelter
            bool is_shelter = false;
            for (int s = 0; s < totalShelters; s++) {
                if (from == d_shelter_city[s]) {
                    is_shelter = true;
                    break;
                }
            }
            
            if (!is_shelter && accumulated_distance <= elderlyDistanceLimit) {
                // This is a non-shelter city within distance limit
                last_non_shelter_before_limit = from;
            }

            if (accumulated_distance <= elderlyDistanceLimit)
                farthest_reachable = to;
            else
                break;
        }

        bool elderly_can_reach = (farthest_reachable == selected_city);
        
        // Attempt to drop elderly at shelter if reachable
        if (elderly_can_reach) {
            if(remaining_elderly > 0)
            {
                int current_capacity;
                int updated_capacity;
                int admitted_elderly;
                bool success = false;

                current_capacity = atomicAdd(&d_shelter_capacity[selected_index], 0);
                admitted_elderly = min(remaining_elderly, (long long)current_capacity);
                updated_capacity = current_capacity - admitted_elderly;

                while (!success && admitted_elderly > 0) {
                    int cas_result = atomicCAS(&d_shelter_capacity[selected_index], current_capacity, updated_capacity);
                    success = (cas_result == current_capacity);
                    if (!success) {
                        current_capacity = atomicAdd(&d_shelter_capacity[selected_index], 0);
                        admitted_elderly = min(remaining_elderly, (long long)current_capacity);
                        updated_capacity = current_capacity - admitted_elderly;
                    }
                }

                remaining_elderly -= admitted_elderly;

                if (admitted_elderly > 0) {
                    res.drops[res.drops_size][0] = selected_city;
                    res.drops[res.drops_size][1] = 0;
                    res.drops[res.drops_size][2] = admitted_elderly;
                    res.drops_size++;
                }

                if (remaining_elderly > 0) {
                    res.drops[res.drops_size][0] = selected_city;
                    res.drops[res.drops_size][1] = 0;
                    res.drops[res.drops_size][2] = remaining_elderly;
                    res.drops_size++;
                    remaining_elderly = 0;
                }
            }
        }
        else if (!elderly_can_reach && remaining_elderly > 0) {
            if (last_non_shelter_before_limit == -1) {
                res.drops[res.drops_size][0] = farthest_reachable;
                res.drops[res.drops_size][1] = 0;
                res.drops[res.drops_size][2] = remaining_elderly;
                res.drops_size++;
            } else {
                res.drops[res.drops_size][0] = last_non_shelter_before_limit;
                res.drops[res.drops_size][1] = 0;
                res.drops[res.drops_size][2] = remaining_elderly;
                res.drops_size++;
            }
            remaining_elderly = 0;
        }

        // Drop remaining prime-age people
        if (remaining_prime > 0) {
            int admitted_prime = 0;

            while (true) {
                int cap_before = atomicAdd(&d_shelter_capacity[selected_index], 0);
                if (cap_before == 0) break;

                int assignable = min(remaining_prime, (long long)cap_before);
                int cap_after = cap_before - assignable;

                if (assignable > 0 &&
                    atomicCAS(&d_shelter_capacity[selected_index], cap_before, cap_after) == cap_before) {
                    admitted_prime = assignable;
                    remaining_prime -= admitted_prime;
                    break;
                }
            }

            if (admitted_prime > 0) {
                res.drops[res.drops_size][0] = selected_city;
                res.drops[res.drops_size][1] = admitted_prime;
                res.drops[res.drops_size][2] = 0;
                res.drops_size++;
            }
        }


        // Update current position
        current_city = selected_city;

        if(no_of_shelters_used >= totalShelters)break;
    }
    
    // If we have any remaining people to evacuate, drop them at the first non-shelter city
    // or at the source city if no non-shelter city was found in the path
    if (remaining_elderly > 0 || remaining_prime > 0) {
        int drop_location = (first_non_shelter_city != -1) ? first_non_shelter_city : source_city;
        // If we still don't have a non-shelter city to drop at, use the current city
        // but double-check it's not a shelter
        if (drop_location == -1) {
            bool is_current_city_shelter = false;
            for (int i = 0; i < totalShelters; i++) {
                if (current_city == d_shelter_city[i]) {
                    is_current_city_shelter = true;
                    break;
                }
            }
            
            // Only use current city if it's not a shelter
            if (!is_current_city_shelter) {
                drop_location = current_city;
            } else {
                // Last resort: if we have no non-shelter city, use source city
                drop_location = source_city;
            }
        }
        
        // Drop remaining elderly
        if (remaining_elderly > 0) {
            res.drops[res.drops_size][0] = drop_location;
            res.drops[res.drops_size][1] = 0;
            res.drops[res.drops_size][2] = remaining_elderly;
            res.drops_size++;
            remaining_elderly = 0;
        }
        
        // Drop remaining prime-age individuals
        if (remaining_prime > 0) {
            long long to_drop = remaining_prime;
            remaining_prime = 0;

            res.drops[res.drops_size][0] = drop_location;
            res.drops[res.drops_size][1] = to_drop;
            res.drops[res.drops_size][2] = 0;
            res.drops_size++;
        }
    }
}

/**
 * Allocates shelter capacity for a given population
 * Returns the number of people successfully assigned to the shelter
 */
__device__ long long allocateShelterCapacity(int *d_shelter_capacity, int shelter_idx, long long population_count) {
    if (population_count <= 0) return 0;
    
    int previous_cap, updated_cap, to_assign;
    int success = 0;
    long long assigned = 0;
    
    do {
        previous_cap = atomicAdd(&d_shelter_capacity[shelter_idx], 0);
        if (previous_cap <= 0) break;
        
        to_assign = min(population_count, (long long)previous_cap);
        updated_cap = previous_cap - to_assign;

        if (to_assign > 0 &&
            atomicCAS(&d_shelter_capacity[shelter_idx], previous_cap, updated_cap) == previous_cap) {
            success = 1;
            assigned = to_assign;
        }
    } while (!success && previous_cap > 0);
    
    return assigned;
}

/**
 * Finds the next unvisited city to move to
 * Returns the selected city index and updates travel cost
 */
__device__ int findNextCity(int current_city, bool *visited, int *d_edge_rowstart, 
                           int *d_edge_count, int *d_edges_target, int *d_edges_len,
                           unsigned int *rng_state, int *travel_cost) {
    int candidate_cities[64];
    int candidate_weights[64];
    int candidate_count = 0;
    
    int start = d_edge_rowstart[current_city];
    int end = start + d_edge_count[current_city];
    
    // Collect all unvisited neighboring cities
    int edge_ptr = start;
    while (edge_ptr < end) {
        if (candidate_count >= 64) break;
        int target_city = d_edges_target[edge_ptr];
        bool is_unvisited = !visited[target_city];
        if (is_unvisited) {
            candidate_weights[candidate_count] = d_edges_len[edge_ptr];
            candidate_cities[candidate_count] = target_city;
            ++candidate_count;
        }
        ++edge_ptr;
    }

    
    // No candidates found
    if (candidate_count == 0) {
        *travel_cost = 0;
        return -1;
    }
    
    // Select a random candidate - FIXED RNG UPDATE
    unsigned int rng_value = (*rng_state * 1664525u + 1013904223u);
    *rng_state = rng_value; // Update the state correctly
    int selected_idx = rng_value % candidate_count;
    
    // Set travel cost for the selected city
    *travel_cost = candidate_weights[selected_idx];
    
    // Return the selected city
    return candidate_cities[selected_idx];
}

__global__ void kernel2(int *d_shelter_capacity, int *d_edges_target, int *d_popcity_elderpop, 
    int totalCities, int *d_edge_count, int *d_populated_city, int *d_edge_rowstart, 
    int totalShelters, int *d_shelter_city, unsigned int seed, int *d_edges_len, 
    int elderlyDistanceLimit, int *d_popcity_primepop, bool *d_visited, 
    int totalPopulated, EvacuationResult *d_results)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= totalPopulated) return;
    
    bool *visited = &d_visited[id * totalCities];
    int source_city = d_populated_city[id];
    int current_city = source_city;
    
    // Initialize visited array correctly
    for (int i = 0; i < totalCities; i++) {
        visited[i] = false;
    }
    visited[current_city] = true;  // Mark current city as visited
    
    unsigned int rng_state = seed + id;
    
    long long remaining_prime = d_popcity_primepop[id];
    long long remaining_elderly = d_popcity_elderpop[id];
    
    EvacuationResult &res = d_results[id];
    res.path_size = 0;
    res.drops_size = 0;

    res.path[0] = current_city;
    res.path_size = 1;

    int backup_drop_city = -1;
    int traveled_elderly = 0;

    // Add an iteration counter to prevent infinite loops
    int iteration_count = 0;
    const int MAX_ITERATIONS = totalCities * 2; // Arbitrary safety limit

    while ((remaining_prime > 0 || remaining_elderly > 0) && iteration_count < MAX_ITERATIONS) {
        iteration_count++;
        
        int shelter_idx = -1;
        int j = 0;
        while (j < totalShelters) {
            if (d_shelter_city[j] != current_city) {
                ++j;
                continue;
            }

            shelter_idx = j;
            break;
        }

        if (shelter_idx < 0) 
        {
            if (backup_drop_city < 0) {
                backup_drop_city = current_city;
            }
        } 
        else 
        {
            // Handle elderly population first using device function
            if (remaining_elderly > 0) {
                long long assigned_elderly = allocateShelterCapacity(d_shelter_capacity, shelter_idx, remaining_elderly);
                
                if (assigned_elderly > 0 && res.drops_size < 1000) {
                    res.drops[res.drops_size][0] = current_city;
                    res.drops[res.drops_size][1] = 0;
                    res.drops[res.drops_size][2] = assigned_elderly;
                    res.drops_size++;
                }
                remaining_elderly -= assigned_elderly;
            }
            
            // Then handle prime-age population using the same device function
            if (remaining_prime > 0) {
                long long assigned_prime = allocateShelterCapacity(d_shelter_capacity, shelter_idx, remaining_prime);
                
                if (assigned_prime > 0 && res.drops_size < 1000) {
                    res.drops[res.drops_size][0] = current_city;
                    res.drops[res.drops_size][1] = assigned_prime;
                    res.drops[res.drops_size][2] = 0;
                    res.drops_size++;
                }
                remaining_prime -= assigned_prime;
            }
        } 

        if (remaining_prime == 0 && remaining_elderly == 0) break;
        
        // Use the device function to find the next city
        int step_cost = 0;
        int next_city = findNextCity(current_city, visited, d_edge_rowstart, 
                                    d_edge_count, d_edges_target, d_edges_len, 
                                    &rng_state, &step_cost);
                                    
        // No valid cities to move to - check if we visited all cities
        if (next_city == -1) {
            break;
        }

        // Evaluate elderly distance constraint
        bool over_limit = (remaining_elderly > 0 && (traveled_elderly + step_cost > elderlyDistanceLimit));
        if (over_limit) {
            if (res.drops_size < 1000) {
                res.drops[res.drops_size][0] = current_city;
                res.drops[res.drops_size][1] = 0;
                res.drops[res.drops_size][2] = remaining_elderly;
                res.drops_size++;
            }
            remaining_elderly = 0;
        }

        visited[current_city] = true;
        current_city = next_city;

        if (remaining_elderly > 0) {
            traveled_elderly += step_cost;
        }

        if (res.path_size >= 1000) {
            break;
        } else {
            res.path[res.path_size++] = current_city;
        }
    }

    if (remaining_prime > 0 && res.drops_size < 1000) {
        int drop_location;
        if (backup_drop_city >= 0) {
            drop_location = backup_drop_city;
        } else {
            drop_location = current_city;
        }
        res.drops[res.drops_size][0] = drop_location;
        res.drops[res.drops_size][1] = remaining_prime;
        res.drops[res.drops_size][2] = 0;
        res.drops_size++;
        remaining_prime = 0;
    }
}
void convertadjToCSR(const vector<vector<Edge>>& graph, 
    vector<int>& destNodes, 
    vector<int>& edgeWeights, 
    vector<int>& nodeStartIdx, 
    vector<int>& nodeDegree) 
{
    int numNodes = graph.size();
    
    destNodes.clear();
    edgeWeights.clear();

    nodeStartIdx.resize(numNodes);
    nodeDegree.resize(numNodes);

    for (int node = 0; node < numNodes; ++node) 
    {
        nodeStartIdx[node] = destNodes.size();
        nodeDegree[node] = graph[node].size();
        for (const Edge& edge : graph[node]) {
            destNodes.push_back(edge.targetNode);
            edgeWeights.push_back(edge.weight);
        }
    }
}

// Function to allocate and copy populated city data to GPU
void allocateAndCopyPopulatedCityData(int num_populated_cities, 
    const vector<int>& populated_city,
    const vector<int>& popcity_primepop,
    const vector<int>& popcity_elderpop,
    int** d_populated_city,
    int** d_popcity_primepop,
    int** d_popcity_elderpop) {
cudaMalloc(d_populated_city, num_populated_cities * sizeof(int));
cudaMemcpy(*d_populated_city, populated_city.data(), num_populated_cities * sizeof(int), cudaMemcpyHostToDevice);

cudaMalloc(d_popcity_primepop, num_populated_cities * sizeof(int));
cudaMemcpy(*d_popcity_primepop, popcity_primepop.data(), num_populated_cities * sizeof(int), cudaMemcpyHostToDevice);

cudaMalloc(d_popcity_elderpop, num_populated_cities * sizeof(int));
cudaMemcpy(*d_popcity_elderpop, popcity_elderpop.data(), num_populated_cities * sizeof(int), cudaMemcpyHostToDevice);
}

// Function to allocate and copy shelter data to GPU
void allocateAndCopyShelterData(int num_shelters,
const vector<int>& shelter_city,
const vector<int>& shelter_capacity,
int** d_shelter_city,
int** d_shelter_capacity) {
cudaMalloc(d_shelter_city, num_shelters * sizeof(int));
cudaMemcpy(*d_shelter_city, shelter_city.data(), num_shelters * sizeof(int), cudaMemcpyHostToDevice);

cudaMalloc(d_shelter_capacity, num_shelters * sizeof(int));
cudaMemcpy(*d_shelter_capacity, shelter_capacity.data(), num_shelters * sizeof(int), cudaMemcpyHostToDevice);
}

// Function to allocate and copy graph data to GPU
void allocateAndCopyGraphData(int num_cities,
const vector<int>& edges_target,
const vector<int>& edges_len,
const vector<int>& edge_rowstart,
const vector<int>& edge_count,
int** d_edges_target,
int** d_edges_len,
int** d_edge_rowstart,
int** d_edge_count) {
cudaMalloc(d_edges_target, edges_target.size() * sizeof(int));
cudaMemcpy(*d_edges_target, edges_target.data(), edges_target.size() * sizeof(int), cudaMemcpyHostToDevice);

cudaMalloc(d_edges_len, edges_len.size() * sizeof(int));
cudaMemcpy(*d_edges_len, edges_len.data(), edges_len.size() * sizeof(int), cudaMemcpyHostToDevice);

cudaMalloc(d_edge_rowstart, num_cities * sizeof(int));
cudaMemcpy(*d_edge_rowstart, edge_rowstart.data(), num_cities * sizeof(int), cudaMemcpyHostToDevice);

cudaMalloc(d_edge_count, num_cities * sizeof(int));
cudaMemcpy(*d_edge_count, edge_count.data(), num_cities * sizeof(int), cudaMemcpyHostToDevice);
}

// Function to free populated city data from GPU
void freePopulatedCityData(int* d_populated_city, int* d_popcity_primepop, int* d_popcity_elderpop) {
cudaFree(d_populated_city);
cudaFree(d_popcity_primepop);
cudaFree(d_popcity_elderpop);
}

// Function to free shelter data from GPU
void freeShelterData(int* d_shelter_city, int* d_shelter_capacity) {
cudaFree(d_shelter_city);
cudaFree(d_shelter_capacity);
}

// Function to free graph data from GPU
void freeGraphData(int* d_edges_target, int* d_edges_len, int* d_edge_rowstart, int* d_edge_count) {
cudaFree(d_edges_target);
cudaFree(d_edges_len);
cudaFree(d_edge_rowstart);
cudaFree(d_edge_count);
}


int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    ifstream infile(argv[1]); // Read input file from command-line argument
    if (!infile) {
        cerr << "Error: Cannot open file " << argv[1] << "\n";
        return 1;
    }

    long long num_cities;
    infile >> num_cities;

    long long num_roads;
    infile >> num_roads;
    
    // Read graph
    vector<vector<Edge>> adj_list(num_cities);
    int i=0;
    while (i < num_roads) 
    {
        int startv,endv;
        infile >> startv >> endv;
        int length,capacity;
        infile >> length >> capacity;
        adj_list[startv].push_back({endv, length});
        adj_list[endv].push_back({startv, length});
        i++;
    }

    // Read shelters
    int num_shelters;
    infile >> num_shelters;
    unordered_map<int, int> city_to_sid;
    vector<int> shelter_city(num_shelters);
    vector<int> shelter_capacity(num_shelters);
    vector<int> original_capacity(num_shelters);
    
    i=0;
    while (i < num_shelters) {
        infile >> shelter_city[i];
        infile >> shelter_capacity[i];
        city_to_sid[shelter_city[i]] = i;
        original_capacity[i] = shelter_capacity[i];
        i++;
    }

    // Read populated cities
    int num_populated_cities;
    infile >> num_populated_cities;
    vector<int> populated_city(num_populated_cities);
    vector<int> popcity_primepop(num_populated_cities);
    vector<int> popcity_elderpop(num_populated_cities);
    
    i=0;
    while (i < num_populated_cities) {
        infile >> populated_city[i];
        infile >> popcity_primepop[i];
        infile >> popcity_elderpop[i];
        i++;
    }
    
    int max_distance_elderly;
    infile >> max_distance_elderly;

    infile.close();

    // Prepare city2sidx for GPU
    vector<int> cityToShelterIndex(num_cities, -1);
    for (auto it = city_to_sid.begin(); it != city_to_sid.end(); ++it) 
    {
        int cityId = it->first;
        int shelterIdx = it->second;
        cityToShelterIndex[cityId] = shelterIdx;
    }

    vector<int> edges_target, edges_len, edge_rowstart, edge_count;
    convertadjToCSR(adj_list, edges_target, edges_len, edge_rowstart, edge_count);
    // Copy results back
    vector<EvacuationResult> results(num_populated_cities);
    vector<int> updatedShelterCapacity(num_shelters);
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_populated_cities + threadsPerBlock - 1) / threadsPerBlock;
    if(num_cities <= 1000) {
        // Prepare CUDA memory for graph
        // Convert adjacency list to CSR format for GPU
        
        // Populated city info
        int *d_populated_city, *d_popcity_primepop, *d_popcity_elderpop;
        allocateAndCopyPopulatedCityData(num_populated_cities, populated_city, 
          popcity_primepop, popcity_elderpop,
          &d_populated_city, &d_popcity_primepop, &d_popcity_elderpop);
        
        // Shelter info
        int *d_shelter_city, *d_shelter_capacity;
        allocateAndCopyShelterData(num_shelters, shelter_city, shelter_capacity,
        &d_shelter_city, &d_shelter_capacity);
        
        // Mapping from city to shelter index
        int *d_cityToShelterIndex;
        cudaMalloc(&d_cityToShelterIndex, num_cities * sizeof(int));
        cudaMemcpy(d_cityToShelterIndex, cityToShelterIndex.data(), num_cities * sizeof(int), cudaMemcpyHostToDevice);
        
        // CSR Graph representation
        int *d_edges_target, *d_edges_len, *d_edge_rowstart, *d_edge_count;
        allocateAndCopyGraphData(num_cities, edges_target, edges_len, edge_rowstart, edge_count,
        &d_edges_target, &d_edges_len, &d_edge_rowstart, &d_edge_count);
        
        // Result buffer
        EvacuationResult* d_results;
        cudaMalloc(&d_results, num_populated_cities * sizeof(EvacuationResult));
        
        // Launch kernel
        kernel1<<<blocksPerGrid, threadsPerBlock>>>(num_cities, num_shelters, num_populated_cities, 
                     d_populated_city, d_popcity_primepop, d_popcity_elderpop,
                     d_shelter_city, d_shelter_capacity, d_cityToShelterIndex, 
                     d_edges_target, d_edges_len, d_edge_rowstart, d_edge_count, 
                     d_results, max_distance_elderly);
        
        cudaDeviceSynchronize();
        
        // === Free some device memory first ===
        cudaFree(d_populated_city);
        cudaFree(d_popcity_primepop);
        
        // === Copy results and capacity back before freeing them ===
        cudaMemcpy(results.data(), d_results, num_populated_cities * sizeof(EvacuationResult), cudaMemcpyDeviceToHost);
        cudaMemcpy(updatedShelterCapacity.data(), d_shelter_capacity, num_shelters * sizeof(int), cudaMemcpyDeviceToHost);
        
        // === Continue freeing the rest ===
        cudaFree(d_popcity_elderpop);
        cudaFree(d_shelter_city);
        cudaFree(d_shelter_capacity);
        cudaFree(d_cityToShelterIndex);
        
        freeGraphData(d_edges_target, d_edges_len, d_edge_rowstart, d_edge_count);
        cudaFree(d_results);
        }
        else {
        // Populated city info
        int *d_populated_city, *d_popcity_primepop, *d_popcity_elderpop;
        allocateAndCopyPopulatedCityData(num_populated_cities, populated_city, 
          popcity_primepop, popcity_elderpop,
          &d_populated_city, &d_popcity_primepop, &d_popcity_elderpop);
        
        // === Shelter data ===
        int *d_shelter_city, *d_shelter_capacity;
        allocateAndCopyShelterData(num_shelters, shelter_city, shelter_capacity,
        &d_shelter_city, &d_shelter_capacity);
        
        // === Graph (CSR) data ===
        int *d_edges_target, *d_edges_len, *d_edge_rowstart, *d_edge_count;
        allocateAndCopyGraphData(num_cities, edges_target, edges_len, edge_rowstart, edge_count,
        &d_edges_target, &d_edges_len, &d_edge_rowstart, &d_edge_count);
        
        // === Output and workspace ===
        EvacuationResult *d_results;
        cudaMalloc(&d_results, num_populated_cities * sizeof(EvacuationResult));
        
        bool *d_visited;
        cudaMalloc(&d_visited, num_populated_cities * num_cities * sizeof(bool));
        
        // Kernel launch
        unsigned int seed = time(NULL);
        
        kernel2<<<blocksPerGrid, threadsPerBlock>>>(d_shelter_capacity, d_edges_target, 
        d_popcity_elderpop, num_cities, d_edge_count, d_populated_city, 
        d_edge_rowstart, num_shelters, d_shelter_city, seed, d_edges_len, 
        max_distance_elderly, d_popcity_primepop, d_visited, 
        num_populated_cities, d_results);
        
        cudaDeviceSynchronize();
        
        // Copy results back
        cudaMemcpy(results.data(), d_results, num_populated_cities * sizeof(EvacuationResult), cudaMemcpyDeviceToHost);
        
        // Copy back updated shelter capacities
        cudaMemcpy(updatedShelterCapacity.data(), d_shelter_capacity, num_shelters * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Free GPU memory
        // === Free output and workspace ===
        cudaFree(d_results);
        cudaFree(d_visited);
        
        // Free all other GPU resources
        freeGraphData(d_edges_target, d_edges_len, d_edge_rowstart, d_edge_count);
        freeShelterData(d_shelter_city, d_shelter_capacity);
        freePopulatedCityData(d_populated_city, d_popcity_primepop, d_popcity_elderpop);
        }

    // Open output file
    ofstream outfile(argv[2]);
    if (!outfile.is_open()) {
        cerr << "Error: Cannot open file " << argv[2] << "\n";
        return 1;
    }

    // Write paths directly from results
    for (int i = 0; i < num_populated_cities; ++i) {
        for (int j = 0; j < results[i].path_size; ++j) {
            outfile << results[i].path[j] << " ";
        }
        outfile << "\n";
    }

    // Write drops directly from results
    for (int i = 0; i < num_populated_cities; ++i) {
        for (int j = 0; j < results[i].drops_size; ++j) {
            for (int k = 0; k < 3; ++k) {
                outfile << results[i].drops[j][k] << " ";
            }
        }
        outfile << "\n";
    }
    
    /**************************************************************end of file*************************************************/
    // Convert device results to structured output format
    // vector<vector<int>> evacuation_paths(num_populated_cities);
    // vector<vector<array<long long, 3>>> evacuation_drops(num_populated_cities);

    // for (int city_idx = 0; city_idx < num_populated_cities; city_idx++) {
    //     evacuation_drops[city_idx].resize(results[city_idx].drops_size);
    //     for (int j = 0; j < results[city_idx].drops_size; j++) {
    //         evacuation_drops[city_idx][j][0] = results[city_idx].drops[j][0];
    //         evacuation_drops[city_idx][j][1] = results[city_idx].drops[j][1];
    //         evacuation_drops[city_idx][j][2] = results[city_idx].drops[j][2];
    //     }

    //     evacuation_paths[city_idx].resize(results[city_idx].path_size);
    //     for (int j = 0; j < results[city_idx].path_size; j++) {
    //         evacuation_paths[city_idx][j] = results[city_idx].path[j];
    //     }
    // }
    // // Calculate total individuals assigned to each shelter and apply penalty if needed
    // vector<long long> shelter_usage(num_shelters, 0);

    // for (int city_idx = 0; city_idx < num_populated_cities; city_idx++) {
    //     for (const auto& drop : evacuation_drops[city_idx]) {
    //         int drop_location = static_cast<int>(drop[0]);
    //         auto search = city_to_sid.find(drop_location);
    //         if (search != city_to_sid.end()) {
    //             int shelter_id = search->second;
    //             shelter_usage[shelter_id] += drop[1] + drop[2];  // prime + elderly
    //         }
    //     }
    // }
    
    // cout << "\n====== SHELTER ACCOMMODATION SUMMARY ======\n";

    // vector<long long> shelter_survivors(num_shelters, 0);

    // for (int shelter_idx = 0; shelter_idx < num_shelters; shelter_idx++) {
    //     long long capacity = original_capacity[shelter_idx];
    //     long long total_dropped = shelter_usage[shelter_idx];
    //     long long overflow = max(0LL, total_dropped - capacity);
    //     long long adjusted_loss = min(2 * overflow, capacity);
    //     long long saved = total_dropped - adjusted_loss;

    //     shelter_survivors[shelter_idx] = saved;

    //     cout << "Shelter " << shelter_city[shelter_idx]
    //         << " → Received: " << total_dropped
    //         << ", Survived: " << saved
    //         << " (Capacity: " << capacity << ")\n";
    // }

    // cout << "===========================================\n";

    // // Compute overall statistics
    // long long total_population = 0, total_survivors = 0;

    // for (int i = 0; i < num_populated_cities; i++) {
    //     total_population += popcity_primepop[i] + popcity_elderpop[i];
    // }

    // for (int i = 0; i < num_shelters; i++) {
    //     total_survivors += shelter_survivors[i];
    // }

    // double survival_rate = total_population > 0 ? (100.0 * total_survivors / total_population) : 0.0;

    // cout << "\n====== OVERALL EVACUATION REPORT ======\n";
    // cout << "Saved: " << total_survivors << " out of " << total_population
    //     << " (" << fixed << setprecision(2) << survival_rate << "%)\n";
    // cout << "=======================================\n";
    
    return 0;
}