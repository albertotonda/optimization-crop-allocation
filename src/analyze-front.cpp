#include <print>
#include "csv.hpp"

extern "C" {

extern int stop_dimension;
double fpli_hv(double *data, int d, int n, const double *ref);

}

template <typename T>
struct std::formatter<std::vector<T>> {
    bool printall{false};
    bool csv{false};

    constexpr auto parse(std::format_parse_context& ctx) {
        auto i = ctx.begin();
        if (i == ctx.end())
            return i;

        if (*i == 'a') {
            printall=true;
            ++i;
        }
        if (*i == 'c') {
            csv = true;
            ++i;
        }

        if (i != ctx.end() && *i != '}')
            throw std::format_error("Invalid format for std::vector<T>");

        return i;
    }

    auto format(const std::vector<T>& v, std::format_context& ctx) const {
        if (csv) {
            bool first = true;
            for (auto && e : v) {
                if(first) {
                    std::format_to(ctx.out(), "{}", e);
                    first = false;
                } else {
                    std::format_to(ctx.out(), ",{}", e);
                }
            }
            return ctx.out();
        }
        std::format_to(ctx.out(), "[ ");
        for (size_t i = 0; i < v.size() && (printall ||i < 30); ++i)
            std::format_to(ctx.out(), "{} ", v[i]);
        if (!printall && v.size() > 30) std::format_to(ctx.out(), "...");
        return std::format_to(ctx.out(), "] ({})", v.size());
    }
};

template<typename T>
T vec_normalize(const T& p, const T& minref, const T& maxref)
{
    const size_t ndims = p.size();
    T rv(ndims, 0.0);
    for (size_t i = 0; i != ndims; ++i)
        rv[i] = (p[i] - minref[i])/(maxref[i] - minref[i]);
    return rv;
}

double compute_hv(const std::vector<std::vector<double>>& front,
                  const std::vector<double>& maxs,
                  const std::vector<double>& mins)
{
    int ndim = front[0].size();
    std::vector<double> flat; // flat and normalized
    std::vector<double> ref(ndim);

    for(int i = 0; i != ndim; ++i)
        ref[i] = 1.0;

    size_t idx{0};
    for (const auto& point : front) {
        auto n = vec_normalize(point, mins, maxs);
        for (size_t i = 0; i != ndim; ++i) {
            flat.push_back(n[i]);
        }
    }

    return fpli_hv(flat.data(), ndim, front.size(), ref.data());
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::println("Usage: {} <objectives csv> <ref point>", argv[0]);
        exit(1);
    }

    auto fn = argv[1];
    std::println("Reading from {}", fn);

    std::vector<std::vector<double>> front;
    std::vector<double> maxs, mins;

    csv::CSVReader reader(fn);
    bool dummy_row{false};
    for (csv::CSVRow& row: reader) {
        if (!dummy_row) {
            dummy_row = true;
            continue;
        }
        std::vector<double> t;
        for (size_t i = 1; i != row.size(); ++i) {
            double val{};
            auto str = row[i].get<std::string_view>();
            auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), val);
            if (ec != std::errc()) {
                std::println("Could not parse row {}, column {}: {}",
                             front.size()+3, i, str);
            }
            t.push_back(val);
            if (maxs.size() < i) {
                maxs.resize(i);
                mins.resize(i);
                maxs[i-1] = val;
                mins[i-1] = val;
            }
            maxs[i-1] = std::max(maxs[i-1], val);
            mins[i-1] = std::min(mins[i-1], val);
        }
        front.push_back(t);
    }

    if (argc > 2) {
        int ndim = maxs.size();
        if (argc != ndim+2) {
            std::println("Need a {}-dimensional reference point", ndim);
            return 1;
        }
        for (int i = 0; i != ndim; ++i) {
            std::string_view arg(argv[2+i]);
            double val;
            auto [ptr, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), val);
            if (ec != std::errc()) {
                std::println("Could not parse reference point dim {}: {}", i+1, arg);
                return 1;
            }
            maxs[i] = val;
        }
    }

    std::println("Read {} lines", front.size());
    std::println("Reference max {} min {}", maxs, mins);
    double hv = compute_hv(front, maxs, mins);
    std::println("Hypervolume = {}", hv);

    return 0;
}
