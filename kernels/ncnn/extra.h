#include <memory>
#include <vector>

namespace ncnn {
class Net;
};

struct NcnnExtra {
  std::shared_ptr<ncnn::Net> net;
  int input_blob_id;
  std::vector<std::vector<int>> state_ids;
  int output_blob_id;
  std::vector<std::vector<int>> output_state_ids;
  NcnnExtra(const std::shared_ptr<ncnn::Net> &net, int input_blob_id,
            const std::vector<std::vector<int>> &state_ids, int output_blob_id,
            const std::vector<std::vector<int>> &output_state_ids)
      : net(net), input_blob_id(input_blob_id), state_ids(state_ids),
        output_blob_id(output_blob_id), output_state_ids(output_state_ids) {}
};
