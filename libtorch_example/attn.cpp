#include <iostream>
#include <torch/torch.h>
#include <ATen/ATen.h>

#include "index_info.h" // helper methods

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


torch::Tensor brute_force(torch::Tensor t1, torch::Tensor t2) {
  // std::cout << "Inputs\n" << t1 << "\n" << t2 << std::endl;
  torch::Tensor out = torch::zeros({t1.size(0) , t2.size(0)});
  // std::cout << t1.size(0) << t1.size(1) << t2.size(0) << t2.size(1);
  for (int i = 0; i < t1.size(0); i++) { // row
    for (int j = 0; j < t2.size(0); j++){  //
      float_t res = 0;
      for (int k = 0; k < t2.size(1); k++) {
        res += (t1[i][k] * t2[j][k]).item<float>();
      }
      out[i][j] = res;
    }
  }
  return out;
}

torch::Tensor brute_force_with_fill(
  torch::Tensor t1, torch::Tensor t2, torch::Tensor mask, float64_t fill
) {
  /* This is the brute force matrix multiplication 

  t1 --> [5, 10]
  t2 --> [5, 10]
  matmul(t1, t2) --> [5, 5]

  mask -> [5, 5]
  fill -> float
  */
  // std::cout << "Inputs\n" << t1 << "\n" << t2 << std::endl;
  torch::Tensor out = torch::zeros({t1.size(0) , t2.size(0)});
  // std::cout << t1.size(0) << t1.size(1) << t2.size(0) << t2.size(1);
  for (int i = 0; i < t1.size(0); i++) { // row
    for (int j = 0; j < t2.size(0); j++){  //
      float_t res = 0;
      for (int k = 0; k < t2.size(1); k++) {
        res += (t1[i][k] * t2[j][k]).item<float>();
      }
      out[i][j] = res;
    }
  }
  return out;
}

// my implementation of scatter add
torch::Tensor scatter_sum_simple(
  torch::Tensor src, torch::Tensor index, int64_t dim
) {
  src = src.contiguous();
  auto sizes = src.sizes().vec();
  sizes[dim] = 1 + *index.max().data_ptr<int64_t>(); // get the target output size
  // std::cout << "Hello World "<< sizes;
  // sizes --> [2, 5]
  torch::Tensor out = torch::empty(sizes, src.options()); // define output

  auto B = 1;
  for (auto i = 0; i < dim; i++)
    B *= src.size(i);
  auto E = src.size(dim);
  auto K = src.numel() / (B * E);
  auto N = out.size(dim);
  // B,E,K,N = 1 5 2 5
  auto index_info = getTensorInfo<int64_t>(index);

  // The AT_DISPATCH_* family of macros provides the ability to
  // conveniently generate specializations of a kernel over all of the
  // dtypes we care about in PyTorch.  We call it "dispatch" because
  // we are "dispatching" to the correct, dtype-specific kernel.
  // read more at:
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "scatter", [&] {
    // AT_DISPATCH_REDUCTION_TYPES // this basically converts string to type

    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    int64_t i, idx;
    out.fill_( (scalar_t)0 );

    // AT_DISPATCH_REDUCTION_TYPES comes here, to understand more about
    // variadic macros:
    // https://docs.microsoft.com/en-us/cpp/preprocessor/variadic-macros
    // in pytorch_scatter it is defined here:
    // https://github.com/rusty1s/pytorch_scatter/blob/master/csrc/cpu/reducer.h
    // so that function parse the string reduce from ReductionType
    // and returns rest of the arguments using __VA_ARGS__()

    for (auto b = 0; b < B; b++) {
      for (auto e = 0; e < E; e++) {
        for (auto k = 0; k < K; k++ ) {
          i = b*E*K + e*K + k;
          idx = index_info.data[IndexToOffset<int64_t>::get(i, index_info)];
          auto *val = out_data + b*N*K + idx*K + k;
          *val = *val + src_data[i];
        }
      }
    }
  });

  return out;
}

// main function
int main (){
  // torch::Tensor t1 = torch::rand({2, 3});
  // torch::Tensor t2 = -torch::rand({2, 3});
  // torch::Tensor out = dot_prod(t1, t2);
  // std::cout << "output:\n" << out << std::endl;

  // std::string mode = std::getenv("MODE");
  // printf("%s", mode.c_str());

  int B = 100;
  int N = 200;
  int H = 2;
  int E = 3000;

  torch::Tensor src = torch::tensor({{2, 0, 1, 4, 3}, {0, 2, 1, 3, 4}});
  torch::Tensor index = torch::tensor({{4, 5, 4, 2, 3}, {0, 0, 2, 2, 1}});
  std::cout << src << "\n" << index;

  torch::Tensor out = scatter_sum_simple(src, index, -1);

  std::cout << "\n" << out << std::endl;

  // brute_force_with_fill
  torch::Tensor t1 = torch::rand({5, 10});
  torch::Tensor t2 = torch::rand({5, 10});
  torch::Tensor mask = torch::tensor({
    {1, 0, 0, 0, 0},
    {1, 1, 0, 0, 0},
    {1, 1, 1, 0, 0},
    {1, 1, 1, 1, 0},
    {1, 1, 1, 1, 1},
  })

  torch::Tensor out = brute_force_with_fill(t1, t2, mask, -10000.);
  std::cout << out;

  // output should be
  // tensor([[0., 0., 4., 3., 3.],
  //         [2., 4., 4., 0., 0.]])

  // torch::Tensor t1 = torch::rand({B, H, N, E / H});
  // torch::Tensor t2 = torch::rand({B, H, N, E / H});

  // torch::Tensor t1 = torch::rand({1, 2});
  // torch::Tensor t2 = torch::rand({2, 3});
  // torch::Tensor t1 = torch::rand({N, E});
  // torch::Tensor t2 = torch::rand({E, B});
  // std::cout << "Inputs\n" << t1 << "\n" << t2 << std::endl;

  // torch::Tensor out = at::matmul(t1, t2);
  // std::cout << out.size(0) << "x" << out.size(1);

  // torch::Tensor t1 = torch::rand({N, E});
  // torch::Tensor t2 = torch::rand({B, E});
  // torch::Tensor out = brute_force(t1, t2);
  // std::cout << out;
}
