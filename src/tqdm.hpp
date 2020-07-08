#pragma once
#if _MSC_VER
#include <io.h>
#define isatty _isatty
#else
#include <unistd.h>
#endif
#include <chrono>
#include <ctime>
#include <numeric>
#include <ios>
#include <string>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <math.h>
#include <algorithm>


class tqdm {
    private:
        // time, iteration counters and deques for rate calculations
        std::chrono::time_point<std::chrono::system_clock> t_first = std::chrono::system_clock::now();
        std::chrono::time_point<std::chrono::system_clock> t_old = std::chrono::system_clock::now();
        int n_old = 0;
        std::vector<double> deq_t;
        std::vector<int> deq_n;
        int nupdates = 0;
        int total_ = 0;
        int period = 1;
        unsigned int smoothing = 50;
        bool use_ema = true;
        float alpha_ema = 0.1;

        std::vector<const char*> bars = {" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};


        bool in_screen = std::getenv("STY") != nullptr;
        bool in_tmux = std::getenv("TMUX") != nullptr;
        bool is_tty = isatty(1);
        bool use_colors = true;
        bool color_transition = true;
        int width = 40;

        std::string right_pad = "▏";
        std::string label = "";

        void hsv_to_rgb(float h, float s, float v, int& r, int& g, int& b);

    public:
        tqdm();

        void reset();

        void set_theme_line() { bars = {"─", "─", "─", "╾", "╾", "╾", "╾", "━", "═"}; }
        void set_theme_circle() { bars = {" ", "◓", "◑", "◒", "◐", "◓", "◑", "◒", "#"}; }
        void set_theme_braille() { bars = {" ", "⡀", "⡄", "⡆", "⡇", "⡏", "⡟", "⡿", "⣿" }; }
        void set_theme_braille_spin() { bars = {" ", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠇", "⠿" }; }
        void set_theme_vertical() { bars = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "█"}; }
        void set_theme_basic() {
            bars = {" ", " ", " ", " ", " ", " ", " ", " ", "#"}; 
            right_pad = "|";
        }

        void set_label(std::string label_) { label = label_; }
        void disable_colors() {
            color_transition = false;
            use_colors = false;
        }

        void finish() {
            progress(total_,total_);
            std::cout << std::endl;
            // printf("\n");
            std::cout << std::flush;
            // fflush(stdout);
        }

        void progress(int curr, int tot);
};