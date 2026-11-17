// Line-oriented TCP connection used for Stratum v1 (one JSON message per line).
#pragma once
#include <mutex>
#include <string>

class StratumConn {
  public:
    ~StratumConn() { close(); }
    bool connect(const std::string& host, int port, std::string& err);
    void close();
    bool connected() const { return fd_ >= 0; }

    // Thread-safe. Appends the trailing newline.
    bool send_line(const std::string& line);
    // Returns 1 when a line was read, 0 on timeout, -1 on error / connection closed.
    int read_line(std::string& line, int timeout_ms);

  private:
    int fd_ = -1;
    std::string buf_;
    std::mutex send_mu_;
};
