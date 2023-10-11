#include <memory>

namespace Ort {
struct Env;
namespace Experimental {
struct Session;
}
} // namespace Ort

struct OnnxExtra {
  std::shared_ptr<Ort::Env> env;
  std::shared_ptr<Ort::Experimental::Session> session;

  OnnxExtra(std::shared_ptr<Ort::Env> env,
            std::shared_ptr<Ort::Experimental::Session> session)
      : env(env), session(session) {}
};
