#include <iostream>
#include <torch/torch.h>

torch::Tensor dot_prod(torch::Tensor t1, torch::Tensor t2, bool verbose = false) {
    if (verbose == true) {
        std::cout << "Inputs\n" << t1 << "\n" << t2 << std::endl;
    }
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

// register
TORCH_LIBRARY(my_ops, m) {
  m.def("dot_prod", dot_prod);
}
