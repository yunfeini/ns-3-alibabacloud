#ifndef CCL_LOG_H
#define CCL_LOG_H

#include <string>
#include <mutex>
#include <fstream>
#include <sstream>
#include "ns3/core-module.h"

namespace ccl {

static std::string g_ccl_log_path;
static std::mutex g_ccl_log_mutex;

inline void SetCclLogPath(const std::string &path) {
  std::lock_guard<std::mutex> lk(g_ccl_log_mutex);
  g_ccl_log_path = path;
}

inline void CclLog(const std::string &msg) {
  std::lock_guard<std::mutex> lk(g_ccl_log_mutex);
  if (g_ccl_log_path.empty())
    return;
  std::ofstream ofs(g_ccl_log_path, std::ofstream::app);
  if (!ofs)
    return;
  std::ostringstream ss;
  ss << "[" << ns3::Simulator::Now().GetNanoSeconds() << "ns] " << msg << "\n";
  ofs << ss.str();
  ofs.flush();
}

} // namespace ccl

#endif // CCL_LOG_H
