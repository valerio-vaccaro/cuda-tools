#include "stratum.h"

#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

bool StratumConn::connect(const std::string& host, int port, std::string& err) {
    close();
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    int rc = getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res);
    if (rc != 0) {
        err = std::string("DNS lookup failed: ") + gai_strerror(rc);
        return false;
    }
    err = "no address to connect to";
    for (addrinfo* ai = res; ai; ai = ai->ai_next) {
        int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        timeval tv{10, 0}; // bounds both connect() and blocking send()
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        if (::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
            fd_ = fd;
            break;
        }
        err = strerror(errno);
        ::close(fd);
    }
    freeaddrinfo(res);
    buf_.clear();
    return fd_ >= 0;
}

void StratumConn::close() {
    std::lock_guard<std::mutex> lk(send_mu_);
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
}

bool StratumConn::send_line(const std::string& line) {
    std::lock_guard<std::mutex> lk(send_mu_);
    if (fd_ < 0) return false;
    std::string data = line + "\n";
    size_t off = 0;
    while (off < data.size()) {
        ssize_t n = ::send(fd_, data.data() + off, data.size() - off, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

int StratumConn::read_line(std::string& line, int timeout_ms) {
    while (true) {
        size_t nl = buf_.find('\n');
        if (nl != std::string::npos) {
            line = buf_.substr(0, nl);
            buf_.erase(0, nl + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return 1;
        }
        if (fd_ < 0 || buf_.size() > (1u << 20)) return -1;
        pollfd pfd{fd_, POLLIN, 0};
        int rc = poll(&pfd, 1, timeout_ms);
        if (rc == 0 || (rc < 0 && errno == EINTR)) return 0;
        if (rc < 0) return -1;
        char tmp[4096];
        ssize_t n = recv(fd_, tmp, sizeof(tmp), 0);
        if (n < 0 && (errno == EINTR || errno == EAGAIN)) return 0;
        if (n <= 0) return -1;
        buf_.append(tmp, (size_t)n);
    }
}
