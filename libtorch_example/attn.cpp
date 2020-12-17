#include <iostream>
#include <torch/torch.h>
#include <ATen/ATen.h>

torch::Tensor dot_prod(torch::Tensor t1, torch::Tensor t2) {
    std::cout << "Inputs\n" << t1 << "\n" << t2 << std::endl;
    torch::Tensor out = torch::zeros({t1.size(0)});
    for (int i = 0; i < t1.size(0); i++) {
        float_t temp = 0;
        for (int j = 0; j < t1.size(1); j++){
            // std::cout << t1[i][j] * t2[i][j];
            temp += (t1[i][j] * t2[i][j]).item<float>();
        }
        out[i] = temp;
    }
    return out;
}

// main function
int main (){ 
    // torch::Tensor t1 = torch::rand({2, 3});
    // torch::Tensor t2 = -torch::rand({2, 3});
    // torch::Tensor out = dot_prod(t1, t2);
    // std::cout << "output:\n" << out << std::endl;

    int B = 1;
    int N = 2;
    int H = 2;
    int E = 4;

    torch::Tensor t1 = torch::rand({B, H, N, E / H});
    torch::Tensor t2 = torch::rand({B, H, N, E / H});

    // torch::Tensor t1 = torch::rand({1, 2});
    // torch::Tensor t2 = torch::rand({2, 3});
    std::cout << "Inputs\n" << t1 << "\n" << t2 << std::endl;
    
    torch::Tensor out = at::matmul(t1, t2);
    std::cout << out;
}
