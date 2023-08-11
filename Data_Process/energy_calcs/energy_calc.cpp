#include <fstream>
#include <iostream>
#include <string>

using namespace std;

float comp_energy(string file){
    std::ifstream read_file(file);
    int n_line = 1;
    string line, sim_time;
    float cpu_time;

    for(int i = 0; i <= n_line; i++){
        getline(read_file, line);
    }
    for (char lett_ : line){
        if (isdigit(lett_) || (lett_ == '.')){
            sim_time += lett_;
        }
    }

    cpu_time = stof(sim_time);
    read_file.close();

    return cpu_time;
}