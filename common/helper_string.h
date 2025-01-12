#pragma once
#include <algorithm>
#include <string>
#include <vector>

using string = std::string;

inline string
lower(string str) {
    std::transform(str.begin(),
                   str.end(),
                   str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return str;
}

inline int
get_cmd_line_arg_int(const std::vector<string>& args, const string& arg_ref, int default_value) {
    for (auto& arg : args) {
        auto arg_end_idx = arg.find_first_of('=');
        if (arg_end_idx == string::npos || arg_end_idx == (arg.size() - 1)) {
            continue;
        } else {
            arg_end_idx -= 1;
        }

        auto arg_start_idx = arg.find_last_of('-');
        arg_start_idx = arg_start_idx == string::npos ? 0 : arg_start_idx + 1;
        auto arg_size = arg_end_idx - arg_start_idx + 1;
        auto arg_ref_size = arg_ref.size();

        if (arg_size == arg_ref_size) {
            int cmp_flag = lower(arg).compare(arg_start_idx, arg_ref_size, lower(arg_ref));

            if (cmp_flag == 0) { return atoi(&arg.c_str()[arg_end_idx + 2]); }
        }
    }
    return default_value;
}

inline string
get_cmd_line_arg_string(const std::vector<string>& args,
                        const string& arg_ref,
                        string default_value) {
    for (auto& arg : args) {
        auto arg_end_idx = arg.find_first_of('=');
        if (arg_end_idx == string::npos || arg_end_idx == (arg.size() - 1)) {
            continue;
        } else {
            arg_end_idx -= 1;
        }

        auto arg_start_idx = arg.find_last_of('-');
        arg_start_idx = arg_start_idx == string::npos ? 0 : arg_start_idx + 1;
        auto arg_size = arg_end_idx - arg_start_idx + 1;
        auto arg_ref_size = arg_ref.size();

        if (arg_size == arg_ref_size) {
            int cmp_flag = lower(arg).compare(arg_start_idx, arg_ref_size, lower(arg_ref));

            if (cmp_flag == 0) { return string{&arg.c_str()[arg_end_idx + 2]}; }
        }
    }
    return default_value;
}