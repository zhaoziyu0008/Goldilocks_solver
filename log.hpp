#ifndef __LOG_HPP
#define __LOG_HPP

#if 1

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>


struct logger_t {
private:
    logger_t() {}
    
    ~logger_t() {}

public:
    static logger_t& instance() {
        static logger_t instance;
        return instance;
    }

    static constexpr int ll_err  = 1;
    static constexpr int ll_warn = 2;
    static constexpr int ll_info = 3;
    static constexpr int ll_dbg  = 4;

    static constexpr int _ll = ll_dbg;

    logger_t(const logger_t&) = delete;
    logger_t& operator=(const logger_t&) = delete;

    void log(int ll, const char* fmt, ...) {
        if (ll > _ll) return;

        char buf[4096];
        va_list args;
        va_start(args, fmt);
        int nbytes = vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);

        if (nbytes < 0 || nbytes >= sizeof(buf)) 
            this->log(ll_warn, "logger_t::log: message truncated");
        
        struct tm tm_info;
        struct timeval tv;
        gettimeofday(&tv, NULL);
        localtime_r(reinterpret_cast<time_t *>(&tv.tv_sec), &tm_info);
        char time_buf[64];

        strftime(time_buf, sizeof(time_buf), "%y-%m-%d %H:%M:%S", &tm_info);
        snprintf(time_buf + strlen(time_buf), sizeof(time_buf) - strlen(time_buf), ".%03ld", tv.tv_usec / 1000);
        
        if (ll == ll_err) {
            fprintf(_log_err, "\033[31m[%s|Err] %s\033[0m\n", time_buf, buf);
        } else if (ll == ll_warn) {
            fprintf(_log_err, "\033[33m[%s|Warn] %s\033[0m\n", time_buf, buf);
        } else if (ll == ll_info) {
            fprintf(_log_out, "\033[36m[%s|Info] %s\033[0m\n", time_buf, buf);
        } else if (ll == ll_dbg) {
            fprintf(_log_out, "\033[0m[%s|Dbg] %s\033[0m\n", time_buf, buf);
        }
        fflush(ll <= ll_warn ? _log_err : _log_out);
    }

    FILE *_log_out = stdout;
    FILE *_log_err = stdout;
};

inline logger_t& logger = logger_t::instance();

#define lg_err(fmt, ...) logger.log(logger_t::ll_err, fmt, ##__VA_ARGS__)
#define lg_warn(fmt, ...) logger.log(logger_t::ll_warn, fmt, ##__VA_ARGS__)
#define lg_info(fmt, ...) logger.log(logger_t::ll_info, fmt, ##__VA_ARGS__)
#define lg_dbg(fmt, ...) logger.log(logger_t::ll_dbg, fmt, ##__VA_ARGS__)

#else

#define lg_err(fmt, ...) 
#define lg_warn(fmt, ...)
#define lg_info(fmt, ...)
#define lg_dbg(fmt, ...)

#endif

#endif