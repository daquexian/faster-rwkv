#include "shape_inference.h"

#include <iostream>

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
  for (int i = nrank - 1, x_idx = x.size() - 1, y_idx = y.size() - 1; i >= 0;
       i--, x_idx--, y_idx--) {
    if (x_idx < 0) {
      output_shape[i] = y[y_idx];
    } else if (y_idx < 0) {
      output_shape[i] = x[x_idx];
    } else if (x[x_idx] == y[y_idx]) {
      output_shape[i] = x[x_idx];
    } else if (x[x_idx] == 1) {
      output_shape[i] = y[y_idx];
    } else if (y[y_idx] == 1) {
      output_shape[i] = x[x_idx];
    } else {
      RV_UNIMPLEMENTED();
    }
  }
  return output_shape;
}

Shape concat(const std::vector<Shape> &xs, int axis) {
  RV_CHECK(xs.size() > 0);
  RV_CHECK(axis >= 0 && axis < xs[0].size());
  Shape output_shape = xs[0];
  for (int i = 1; i < xs.size(); i++) {
    RV_CHECK(xs[i].size() == output_shape.size());
    for (int j = 0; j < xs[i].size(); j++) {
      if (j == axis) {
        output_shape[j] += xs[i][j];
      } else {
        RV_CHECK(xs[i][j] == output_shape[j]);
      }
    }
  }
  return output_shape;
}

Shape slice(const Shape &x, const std::vector<int> &starts,
    const std::vector<int> &ends, const std::vector<int> &axes) {
  RV_CHECK(starts.size() == ends.size());
  RV_CHECK(starts.size() == axes.size());
  Shape output_shape = x;
  for (int i = 0; i < starts.size(); i++) {
    RV_CHECK(starts[i] >= 0 && starts[i] < x[axes[i]]);
    RV_CHECK(ends[i] >= 0 && ends[i] <= x[axes[i]]);
    RV_CHECK(starts[i] < ends[i]);
    output_shape[axes[i]] = ends[i] - starts[i];
  }
  return output_shape;
}

} // namespace shape
} // namespace rwkv
