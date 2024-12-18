#include <print>
#include <algorithm>
#include <ranges>
#include <vector>
#include "csv.hpp"

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

using point = std::vector<double>;

struct pfront {
    std::vector<point> points;
};

std::vector<point> find_dominators(point const& p, pfront const& front)
{
    std::vector<point> rv;
    for(point const& o : front.points) {
        if (&o == &p)
            continue;
        if (std::ranges::none_of(std::ranges::views::zip(p, o), [&](auto t) {
            auto [pd, od] = t;
            return pd > od;
        }))
            rv.push_back(o);
    }
    return rv;
}


int find_dominated(pfront const &front)
{
    int ndomed{0};
    for (point const& p : front.points) {
        auto doms = find_dominators(p, front);
        if (!doms.empty()) {
            ++ndomed;
            std::println("Point {} is dominated by\n\t{}", p, doms);
        }
    }
    return ndomed;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::println("Usage: {} <objectives csv>", argv[0]);
        exit(1);
    }

    auto fn = argv[1];
    std::println("Reading from {}", fn);

    pfront myfront;
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
                             myfront.points.size()+3, i, str);
            }
            t.push_back(val);
        }
        myfront.points.push_back(t);
    }

    std::println("Read {} lines", myfront.points.size());
    int ndomed = find_dominated(myfront);
    std::println("{}/{} points dominated, {} remain", ndomed,
                 myfront.points.size(), myfront.points.size() - ndomed);

    return 0;
}
