#include <iostream>
#include <torch/torch.h>

void dot_prod( torch::Tensor t1, torch::Tensor t2) {
    std::cout << "Hello World\n" << t1 << "\n" << t2 << std::endl;
    torch::Tensor out = torch::zeros({t1.size(0)});
    for (int i = 0; i < t1.size(0); i++) {
        float_t temp = 0;
        for (int j = 0; j < t1.size(1); j++){
            // std::cout << t1[i][j] * t2[i][j];
            temp += (t1[i][j] * t2[i][j]).item<float>();
        }
        out[i] = temp;
    }
    std::cout << out << std::endl;
}

// main function
int main (){
    torch::Tensor t1 = torch::rand({2, 3});
    torch::Tensor t2 = -torch::rand({2, 3});
    dot_prod(t1, t2);
    // std::cout << tensor << std::endl;
}
