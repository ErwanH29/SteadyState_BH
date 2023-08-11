#include "global.h"
#include "energy_calc.h"

#include <cmath>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

using namespace std;

int main(){
    string folder[3] = {"rc_0.25_4e5", "rc_0.25_4e6", "rc_0.25_4e7"};
    float total_cpu = 0;
    float total_con, total_emi;
    int drange;

    for (string fold_ : folder){
        cout << "Analysing " << fold_ << endl;
        if (fold_ == "rc_0.25_4e6"){
            drange = 2;
            int frange[2] = {160, 420};
            string integ[2] = {"Hermite", "GRX"};

            for (int i = 0; i < drange; i++){
                int iter = 0;
                string path = "/media/erwanh/Elements/"+fold_+"/data/"+integ[i]+"/simulation_stats/";
                for (const auto & entry : std::filesystem::directory_iterator(path)){
                    if (iter < frange[i]){
                        total_cpu += comp_energy(entry.path().c_str());
                        cout << "Iteration: " << iter << "File " << entry << " usage: " << total_cpu << endl;
                    }
                    iter++;
                }
            }
        }
        else{
            drange = 1;
            int frange[2] = {120, 120};
            string integ[1] = {"GRX"};

            for (int i = 0; i < drange; i++){
                int iter = 0;
                string path = "/media/erwanh/Elements/"+fold_+"/data/"+integ[i]+"/simulation_stats/";
                for (const auto & entry : std::filesystem::directory_iterator(path)){
                    if (iter < frange[i]){
                        total_cpu += comp_energy(entry.path().c_str());
                        cout << "Iteration: " << iter << ". File " << entry << ". Usage: " << total_cpu << endl;
                    }
                    iter++;
                }
            }
        }
    }

    total_cpu /= hr_secs;
    total_con = total_cpu * no_cpu * cpu_con;
    total_emi = pow(10,-3)*total_con/emiss;

    cout << "Total CPU wall-clock time [hr]: " << total_cpu << "(" << total_cpu/24 << "days, " << total_cpu/24 * 18 << " days)" << endl;
    cout << "Total energy consumed [W]:      " << total_con << endl;
    cout << "Total CO2 emitted [kg]:         " << total_emi << endl;

}