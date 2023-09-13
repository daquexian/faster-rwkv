#include "shape_inference.h"

namespace rwkv {
namespace shape {
Shape matmul(const Shape &x, const Shape &y) {
  int batch, m, n, k;
  const int a_ranks = x.size();
  const int b_ranks = y.size();
  Shape output_shape;
  if (a_ranks == 3 && b_ranks == 3) {
    batch = x[0];
    RV_CHECK(batch == y[0]);
    m = x[1];
    k = x[2];
    RV_CHECK(k == y[1]);
    n = y[2];
    output_shape = {batch, m, n};
  } else {
    RV_CHECK(a_ranks <= 2 && b_ranks <= 2);
    if (a_ranks == 1) {
      RV_CHECK(b_ranks == 2);
      m = 1;
      k = x[0];
      n = y[1];
      output_shape = {n};
    } else if (a_ranks == 2) {
      m = x[0];
      k = x[1];
      if (b_ranks == 1) {
        RV_CHECK(a_ranks == 2);
        RV_CHECK(k == y[0]);
        n = 1;
        output_shape = {m};
      } else {
        RV_CHECK(k == y[0]);
        n = y[1];
        output_shape = {m, n};
      }
    }
  }
  return output_shape;
}

Shape broadcast_binary(const Shape &x, const Shape &y) {
  auto nrank = std::max(x.size(), y.size());
  Shape output_shape(nrank);
  for (int i = nrank - 1; i >= 0; i--) {
    if (i >= x.size()) {
      output_shape[i] = y[i];
    } else if (i >= y.size()) {
      output_shape[i] = x[i];
    } else if (x[i] == y[i]) {
      output_shape[i] = x[i];
    } else if (x[i] == 1) {
      output_shape[i] = y[i];
    } else if (y[i] == 1) {
      output_shape[i] = x[i];
    } else {
      RV_UNIMPLEMENTED();
    }
  }
  return output_shape;
}
} // namespace shape
} // namespace rwkv
