#include "global.h"

#include <fstream>
#include <iostream>
#include <string>

using namespace std;

Outputs comp_energy(string file){

    std::ifstream read_file(file);
    int n_line = 1;
    string line, sim_time_str;
    float cpu_time, wall_clock_time;

    for(int i = 0; i <= n_line; i++){
        getline(read_file, line);
    }
    for (char lett_ : line){
        if (isdigit(lett_) || (lett_ == '.')){
            sim_time_str += lett_;
        }
    }
    read_file.close();

    if (file.find("rc_0.25_4e5") != string::npos){
        cpu_time = stof(sim_time_str)*no_cpu_med;
    }
    else{
        cpu_time = stof(sim_time_str)*no_cpu_long;
    }
    wall_clock_time = stof(sim_time_str);

    Outputs results;
    results.cpu_time = cpu_time;
    results.wall_clock_time = wall_clock_time;

    return results;
}