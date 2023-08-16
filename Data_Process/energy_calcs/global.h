#ifndef GLOBAL_H
#define GLOBAL_H

inline float hr_secs = 3600;
inline float cpu_con = 12;  //Watt/hr 
inline float emiss = 0.283; //kWh/kg
inline float no_cpu_long = 18;
inline float no_cpu_med = 6;

struct Outputs{
    float cpu_time, wall_clock_time;
};

#endif