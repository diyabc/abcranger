#include <iostream>
#include "threadpool.hpp"
#include "a.hpp"

int main(int argc, char **argv)
{
    Test::a = 0;
    ThreadPool::ParallelFor(0, 100, [&] (int i){
        std::thread::id myid = std::this_thread::get_id();
        Test::m.lock();
        std::cout << myid << " -> " << i << " -> " << Test::a << " rand : " << Test::mt() << std::endl;
        Test::m.unlock();
        Test::a++;
    });
    return 0;
}
