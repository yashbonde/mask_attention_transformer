#include <iostream>
#include <torch/torch.h>

int main (){
    torch::Tensor tensor = torch::rand({5, 19});
    std::cout << tensor << std::endl;
}
