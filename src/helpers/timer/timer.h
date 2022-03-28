#pragma once

class Timer
{
private:
    std::chrono::time_point<std::chrono::system_clock> m_start;
    std::string m_region_name;
public:
    Timer(std::string _region_name = "unknown"): m_start(std::chrono::high_resolution_clock::now()),
                m_region_name(_region_name) {}

    double get_time_s()
    {
        auto m_end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::seconds>(m_end - m_start).count();
    }

    double get_time_ms()
    {
        auto m_end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(m_end - m_start).count();
    }

    ~Timer()
    {
        auto m_end = std::chrono::high_resolution_clock::now();
        std::cout << "Region " << m_region_name << " took " <<
            std::chrono::duration_cast<std::chrono::milliseconds>(m_end - m_start).count() << " ms." << std::endl;
    }
};