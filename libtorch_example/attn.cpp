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

// my implementation of scatter add
std::tuple<torch::Tensor>
scatter_simple(
  torch::Tensor src, torch::Tensor index,
  int64_t dim, std::string reduce
) {
  src = src.contiguous();
  auto sizes = src.sizes().vec();
  if (index.numel() == 0) sizes[dim] = 0;
  else sizes[dim] = 1 + *index.max().data_ptr<int64_t>();
  torch::Tensor out = torch::empty(sizes, src.options()); // define output

  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;

  auto B = 1;
  for (auto i = 0; i < dim, i ++)
    B *= src.size(i);
  auto E = src.size(dim);
  auto K = src.numel() / (B * E);
  auto N = out.size(dim);

  auto index_info = getTensorInfo<int64_t>(index);

  // The AT_DISPATCH_* family of macros provides the ability to
  // conveniently generate specializations of a kernel over all of the
  // dtypes we care about in PyTorch.  We call it "dispatch" because
  // we are "dispatching" to the correct, dtype-specific kernel.
  // read more at:
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h
  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "scatter", [&]{
    // AT_DISPATCH_REDUCTION_TYPES // this basically converts string to type
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    int64_t i, idx;

    for (auto b = 0; b < B; b++) {
      for (auto e = 0; e < E; e++) {
        for (auto k = 0; k < K; k++ ) {
          idx = index_info[IndexToOffset<int64_t>::get(i, index_info)];
          // out[out_data + b*N*K + idx*K + k] = src_data[i];
          *(out_data + b*N*K + idx*K + k) = src_data[i];
          *(arg_out_data + b*N*K + idx*K + k) = e;
        }
      }
    }
  })
}



// main function
int main (){
  // torch::Tensor t1 = torch::rand({2, 3});
  // torch::Tensor t2 = -torch::rand({2, 3});
  // torch::Tensor out = dot_prod(t1, t2);
  // std::cout << "output:\n" << out << std::endl;

  int B = 100;
  int N = 200;
  int H = 2;
  int E = 3000;

  // torch::Tensor t1 = torch::rand({B, H, N, E / H});
  // torch::Tensor t2 = torch::rand({B, H, N, E / H});

  // torch::Tensor t1 = torch::rand({1, 2});
  // torch::Tensor t2 = torch::rand({2, 3});
  torch::Tensor t1 = torch::rand({N, E});
  torch::Tensor t2 = torch::rand({E, B});
  // std::cout << "Inputs\n" << t1 << "\n" << t2 << std::endl;

  torch::Tensor out = at::matmul(t1, t2);
  std::cout << out.size(0) << "x" << out.size(1);

  // torch::Tensor t1 = torch::rand({N, E});
  // torch::Tensor t2 = torch::rand({B, E});
  // torch::Tensor out = brute_force(t1, t2);
  // std::cout << out;
}
