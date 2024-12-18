#include <print>
#include <string>
#include <list>
#include <vector>
#include <queue>
#include <set>
#include <memory>
#include <ranges>
#include <algorithm>
#include <random>
#include <tuple>
#include <cassert>
#include <stdexcept>

#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

#include <ilcplex/cplexx.h>

#include "libqhullcpp/RboxPoints.h"
#include "libqhullcpp/Qhull.h"

extern "C" {

extern int stop_dimension;
double fpli_hv(double *data, int d, int n, const double *ref);

}

namespace qh = orgQhull;

double cpuTime(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1000000;
}

//////////////////////////////////////////////////////////////////////
// PARAMETERS
//////////////////////////////////////////////////////////////////////

static const int nsamples{30};

struct prob_config {
    enum coeff_type {
        MINAREA, FIXED, IGNORE
    };
    std::vector<coeff_type> types;
    std::vector<double> coeffs; // ignored if coeff_type is MINAREA

    std::vector<bool> hasbounds;
    std::vector<double> lbound;
    std::vector<double> ubound;
};

// this assumes that the order of problems we are given is area, prod,
// stdd. It should be eventually be changed to be specified on the
// command line.

// Bounds suggested by Mathilde:
// - 20% self-sufficiency: 7260000 tonnes -> regarder entre [7 020 000 ; 7 500 000]
// tonnes
// - 50% self-sufficiency:  18150000 tonnes -> regarder entre [17 250 000;
// 19 050 000] tonnes
// - 75% self-sufficiency: 27225000 tonnes -> regarder entre [26 450 000 ;
// 28 000 000] tonnes
// J'ai l'impression que ça fait des gammes de 5% à peu près entre chaque
// objectif de production.

prob_config global_config {
    { prob_config::MINAREA,
      prob_config::MINAREA,
      prob_config::MINAREA },
    { 1.0, 1.0, 0.1 },
    { false, false, false }
};

prob_config config_stages[] = {
    { { prob_config::MINAREA,
          prob_config::MINAREA },
      { 0.001, 1.0 },
      { false, true, false},
      { 0, -7'500'000, 0},
      { 0, -7'020'000, 0}
    },
    { { prob_config::IGNORE,
          prob_config::MINAREA,
          prob_config::MINAREA },
      { 0.001, 1.0, 1.0 },
      { false, true, false},
      { 0, -7'500'000, 0},
      { 0, -7'020'000, 0}
    }
};

// prob_config global_config {
//     { prob_config::MINAREA,
//       prob_config::MINAREA },
//     { 0.0, 0.0 }
// };

bool project_move{true};

//////////////////////////////////////////////////////////////////////
// END PARAMETERS
//////////////////////////////////////////////////////////////////////

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

class cplexenv;

template<typename T>
void throw_if_null(T* p, cplexenv& env, int status, std::string_view msg);

class cplexenv
{
public:
    CPXENVptr env{nullptr};
    std::random_device dev;
    std::mt19937 rng{dev()};

    cplexenv() {
        int status;
        env = CPXXopenCPLEX(&status);
        throw_if_null(env, *this, status, "Could not open CPLEX environment");
        //CPXXsetintparam(env, CPX_PARAM_SCRIND, CPX_ON);
        CPXXsetdblparam(env, CPX_PARAM_BAREPCOMP, 1e-4);
        //CPXXsetdblparam(env, CPX_PARAM_EPOPT, 1e-4);
    }
};

template<typename T>
void throw_if_null(T* p, cplexenv& cpx, int status, std::string_view msg)
{
    if (!p) {
        char errmsg[CPXMESSAGEBUFSIZE];
        auto errstr = CPXXgeterrorstring(cpx.env, status, errmsg);
        if (errstr)
            throw std::runtime_error(std::format("{}, CPLEX error id {}: {}",
                                                 msg.data(), status, errmsg));
        else
            throw std::runtime_error(std::format("{}, CPLEX error id {}",
                                                 msg.data(), status));
    }
}

class lpprob
{
public:
    cplexenv& cpx;
    CPXLPptr lp{nullptr};
    int type;
    std::string readfrom;

    lpprob() = delete;
    explicit lpprob(cplexenv& env) : cpx(env) {
        int status;
        lp = CPXXcreateprob(cpx.env, &status, "prob");
        throw_if_null(lp, cpx, status, "Could not create CPLEX LP problem");
    }
    ~lpprob() {
        CPXXfreeprob(cpx.env, &lp);
    }

    void read(std::string fn)
    {
        readfrom = fn;
        CPXXreadcopyprob(cpx.env, lp, fn.c_str(), "LP");
        type = CPXXgetprobtype(cpx.env, lp);
        switch(type) {
        case CPXPROB_LP:
            std::println("{} is an LP", fn);
            break;
        case CPXPROB_QP:
            std::println("{} is a QP", fn);
            break;
        default:
            throw std::runtime_error(std::format("{}: Unsupported problem type", fn));
        };
        CPXXwriteprob(cpx.env, lp, std::format("rewritten-{}", fn).c_str(), nullptr);
    }

    void solve()
    {
        std::println("Solving \"{}\"", readfrom);
        //CPXXwriteprob(cpx.env, lp, "out.lp", nullptr);
        if (type == CPXPROB_LP) {
            int r = CPXXlpopt(cpx.env, lp);
            if (r)
                throw std::runtime_error("Error solving LP");
        } else if (type == CPXPROB_QP) {
            int r = CPXXqpopt(cpx.env, lp);
            if (r)
                throw std::runtime_error("Error solving QP");
        }  else {
            throw std::runtime_error("No problem type");
        }
    }

    void solve_or_perturb()
    {
        try {
            solve();
        } catch(std::exception& e) {
            std::println("Failed to solve, retrying with perturbation");
            perturb_and_solve();
        }
    }

    void perturb_and_solve()
    {
        bool keeptrying{true};
        double noise = 1e-3;
        while (keeptrying) {
            random_perturb(noise);
            try {
                solve();
                keeptrying = false;
            } catch(std::exception& e) {
                std::println("Failed, retrying");
                noise = noise * 10.0;
            }
        }
    }

    void random_perturb(double noise)
    {
        switch(type) {
        case CPXPROB_LP:
            random_perturb_lp(noise);
            break;
        case CPXPROB_QP:
            random_perturb_qp(noise);
            break;
        default:
            throw std::runtime_error(std::format("{}: Unsupported problem type", type));
        }
    }

    void random_perturb_lp(double)
    {
        throw std::runtime_error("LPs should never need perturbation");
    }

    void random_perturb_qp(double noise)
    {
        std::uniform_real_distribution<> dist(-noise, noise);

        std::println("Perturbing QP, noise = {}", noise);

        int status;
        CPXLPptr newlp = CPXXcreateprob(cpx.env, &status, "prob");
        throw_if_null(lp, cpx, status, "Could not create perturbed CPLEX LP problem");

        int ncols = CPXXgetnumcols(cpx.env, lp);
        std::vector<double> lb(ncols), ub(ncols);
        CPXXgetlb(cpx.env, lp, lb.data(), 0, ncols-1);
        CPXXgetub(cpx.env, lp, ub.data(), 0, ncols-1);
        // fumble to get iota over doubles
        auto coeff = std::ranges::views::repeat(0.0, ncols)
            | std::ranges::to<std::vector>();
        CPXXnewcols(cpx.env, newlp, ncols, coeff.data(),
                    lb.data(), ub.data(), nullptr, nullptr);

        for (int i = 0; i < ncols; ++i) {
            for(int j = i; j < ncols; ++j) {
                double c, e;
                CPXXgetqpcoef(cpx.env, lp, i, j, &c);
                e = c + dist(cpx.rng);
                CPXXchgqpcoef(cpx.env, newlp, i, j, e);
            }
        }

        CPXXfreeprob(cpx.env, &lp);
        lp = newlp;
    }

    std::vector<double> solution()
    {
        int ncols = CPXXgetnumcols(cpx.env, lp);
        std::vector<double> sol(ncols);
        CPXXgetx(cpx.env, lp, sol.data(), 0, ncols-1);
        return sol;
    }

    double solution_cost()
    {
        double val;
        CPXXgetobjval(cpx.env, lp, &val);
        return val;
    }

    double evaluate(const std::vector<double>& sol)
    {
        double rv{0.0};
        int ncols = CPXXgetnumcols(cpx.env, lp);
        if(type == CPXPROB_LP) {
            std::vector<double> coeff(ncols);
            CPXXgetobj(cpx.env, lp, coeff.data(), 0, ncols-1);
            for (int i = 0; i != ncols; ++i)
                rv += coeff[i]*sol[i];
        } else if (type == CPXPROB_QP) {
            for (int i = 0; i != ncols; ++i)
                for(int j = i; j != ncols; ++j) {
                    double c;
                    CPXXgetqpcoef(cpx.env, lp, i, j, &c);
                    rv += .5*c*sol[i]*sol[j];
                }
            // hack, but in our problem we know that the true objective of
            // the problem in the QP case is actually sqrt(obj)
            if (type == CPXPROB_QP)
                rv = std::sqrt(rv);
        }
        return rv;
    }
};

std::unique_ptr<lpprob> make_scaled(cplexenv& cpx,
                                    const std::vector<std::unique_ptr<lpprob>>& probs,
                                    const std::vector<double>& weights)
{
    std::println("Scaling with non-normalized weights {}", weights);
    assert(!probs.empty() && probs.size() == weights.size());

    // normalize weights
    double sumweights = std::ranges::fold_left(weights, 0.0, std::plus<double>{});
    auto normalize = [&](auto&& x) { return x/sumweights; };
    auto norm_w = weights
        | std::ranges::views::transform(normalize)
        | std::ranges::to<std::vector>();

    auto rv = std::make_unique<lpprob>(cpx);
    rv->readfrom = std::format("Scaled problem, weights = {}", norm_w);
    rv->type = CPXPROB_LP;

    bool created_cols{false};
    for (auto&& [pptr, weight] : std::ranges::views::zip(probs, norm_w)) {
        auto& lp = *pptr;
        int ncols = CPXXgetnumcols(cpx.env, lp.lp);
        if(lp.type == CPXPROB_LP) {
            std::vector<double> coeff(ncols);
            CPXXgetobj(cpx.env, lp.lp, coeff.data(), 0, ncols-1);

            if (!created_cols) {
                std::vector<double> lb(ncols), ub(ncols);
                CPXXgetlb(cpx.env, lp.lp, lb.data(), 0, ncols-1);
                CPXXgetub(cpx.env, lp.lp, ub.data(), 0, ncols-1);
                auto scale = [&](auto&& x) { return x * weight; };
                auto scaled_c = coeff
                    | std::ranges::views::transform(scale)
                    | std::ranges::to<std::vector>();
                CPXXnewcols(cpx.env, rv->lp, ncols, scaled_c.data(),
                            lb.data(), ub.data(), nullptr, nullptr);
                created_cols = true;
            } else {
                std::vector<double> oldobj(ncols);
                CPXXgetobj(cpx.env, rv->lp, oldobj.data(), 0, ncols-1);
                auto combine = [&](double x, double y) { return x + y * weight; };
                auto new_c = std::ranges::views::zip_transform(combine, oldobj, coeff)
                    | std::ranges::to<std::vector>();
                auto indices = std::ranges::views::iota(0, ncols)
                    | std::ranges::to<std::vector>();
                CPXXchgobj(cpx.env, rv->lp, ncols, indices.data(), new_c.data());
            }
        } else if(lp.type == CPXPROB_QP) {
            auto prevtype = rv->type;
            rv->type = CPXPROB_QP;
            if (!created_cols) {
                // same as above, but default to 0
                std::vector<double> lb(ncols), ub(ncols);
                CPXXgetlb(cpx.env, lp.lp, lb.data(), 0, ncols-1);
                CPXXgetub(cpx.env, lp.lp, ub.data(), 0, ncols-1);
                // fumble to get iota over doubles
                auto coeff = std::ranges::views::repeat(0.0, ncols)
                    | std::ranges::to<std::vector>();
                CPXXnewcols(cpx.env, rv->lp, ncols, coeff.data(),
                            lb.data(), ub.data(), nullptr, nullptr);
                created_cols = true;
            }
            int nchg{0};
            for (int i = 0; i < ncols; ++i)
                for(int j = i; j < ncols; ++j) {
                    double c, e;
                    CPXXgetqpcoef(cpx.env, lp.lp, i, j, &c);
                    if (prevtype == CPXPROB_QP)
                        CPXXgetqpcoef(cpx.env, rv->lp, i, j, &e);
                    else
                        e = 0.0;
                    e = e + c * weight;
                    CPXXchgqpcoef(cpx.env, rv->lp, i, j, e);
                    ++nchg;
                }
        } else {
            throw std::runtime_error(std::format("Unsupported problem type {}",
                                                 lp.type));
        }
    }
    return rv;
}

using performance_v = std::vector<double>;
using weight_v = std::vector<double>;
struct extremes {
    std::vector<performance_v> extremes;
    performance_v min_observed, max_observed;
};

// returns the performances when optimizing each single objective, as
// well as tuples of the minimum and maximum values for the
// objectives
extremes get_extremes(cplexenv& cpx,
                      std::vector<std::unique_ptr<lpprob>> const& probs,
                      FILE *objectives,
                      FILE *solutions)
{
    std::println("Solving single-objective subproblems");

    std::vector<performance_v> all_sobj;
    performance_v min_observed, max_observed;
    for(auto && p : probs) {
        p->solve();
        auto psol = p->solution();
        std::println("Solved {}, cost = {}", p->readfrom, p->solution_cost());
        auto pfpoint = probs
            | std::ranges::views::transform([&](auto&& q){
                return q->evaluate(psol); })
            | std::ranges::to<std::vector>();
        all_sobj.push_back(pfpoint);
        std::println("Point {}", pfpoint);
        std::println(objectives, "{:c}", pfpoint);
        std::println(solutions, "{:c}", psol);
        if (min_observed.empty()) {
            min_observed = pfpoint;
            max_observed = pfpoint;
        } else {
            auto tmp = std::views::zip_transform([&](double a, double b) {
                return std::min(a,b); },
                min_observed, pfpoint)
                | std::ranges::to<std::vector>();
            std::swap(tmp, min_observed);
            tmp = std::views::zip_transform([&](double a, double b) {
                return std::max(a,b); },
                max_observed, pfpoint)
                | std::ranges::to<std::vector>();
            std::swap(tmp, max_observed);
        }
    }

    std::println("Max observed {}\nMin observed {}",
                 max_observed, min_observed);

    return {all_sobj, min_observed, max_observed};
}

void explore_pareto_weighted_uniform(cplexenv& cpx,
                                     std::vector<std::unique_ptr<lpprob>> const& probs,
                                     FILE *objectives,
                                     FILE *solutions)
{
    std::ignore = get_extremes(cpx, probs, objectives, solutions);
    std::println("Solving single-objective subproblems");
    for(auto && p : probs) {
        p->solve();
        auto psol = p->solution();
        std::println("Solved {}, cost = {}", p->readfrom, p->solution_cost());
        auto pfpoint = probs
            | std::ranges::views::transform([&](auto&& q){
                return q->evaluate(psol); })
            | std::ranges::to<std::vector>();
        std::println("Point {}", pfpoint);
        std::println(objectives, "{:c}", pfpoint);
        std::println(solutions, "{:c}", psol);
    }

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(1,100);

    for(auto sampleid : std::ranges::views::iota(1, nsamples)) {
        std::println("Getting sample {}", sampleid);
        std::vector<double> weights(probs.size());
        // random weights, will be normalized by make_scaled
        for(size_t i = 0; i != probs.size(); ++i)
            weights[i] = dist(rng);

        auto scaled = make_scaled(cpx, probs, weights);
        try {
            scaled->solve();
        } catch(std::exception& e) {
            std::print("Abort attempt, runtime error {}", e.what());
            continue;
        }
        auto psol = scaled->solution();
        std::println("Scaled optimum {}", scaled->solution_cost());
        auto pfpoint = probs
            | std::ranges::views::transform([&](auto&& q){
                return q->evaluate(psol); })
            | std::ranges::to<std::vector>();
        std::println("Point {}", pfpoint);
        std::println(objectives, "{:c}", pfpoint);
        std::println(solutions, "{:c}", psol);
    }
}

void explore_pareto_weighted_adaptive(cplexenv& cpx,
                                      std::vector<std::unique_ptr<lpprob>> const& probs,
                                      FILE *objectives,
                                      FILE *solutions)
{
    auto [all_sobj,
          min_observed, max_observed] = get_extremes(cpx, probs, objectives,
                                                     solutions);

    std::vector<double> adjustment(probs.size(), 1.0);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(1,100);

    for(auto sampleid : std::ranges::views::iota(1, nsamples)) {
        std::println("Getting sample {}", sampleid);
        std::println("Adjustment {}", adjustment);
        weight_v weights(probs.size());
        // random weights, will be normalized by make_scaled
        for(size_t i = 0; i != probs.size(); ++i)
            weights[i] = dist(rng)*adjustment[i];
        weights[1] *= 0.001;

        auto scaled = make_scaled(cpx, probs, weights);
        try {
            scaled->solve();
        } catch(std::exception& e) {
            std::println("Abort attempt, runtime error {}", e.what());
            continue;
        }
        auto psol = scaled->solution();
        std::println("Scaled optimum {}", scaled->solution_cost());
        auto pfpoint = probs
            | std::ranges::views::transform([&](auto&& q){
                return q->evaluate(psol); })
            | std::ranges::to<std::vector>();
        std::println("Point {}", pfpoint);
        std::println(objectives, "{:c}", pfpoint);
        std::println(solutions, "{:c}", psol);

        for(size_t i = 0; i != probs.size(); ++i) {
            double x = probs[i]->evaluate(psol);
            double adj = (x-min_observed[i])
                / (max_observed[i]-min_observed[i]);
            std::println("Adjusting {} by {}", probs[i]->readfrom, adj);
            adjustment[i] += adj;
        }
    }
}

template<int N>
struct vec {
    double p[N];

    vec() {}
    explicit vec(size_t s, double init = 0.0) {
        assert(s == N);
        for (int i = 0; i != N; ++i)
            p[i] = init;
    }
    explicit vec(std::vector<double> const& v) {
        assert(v.size() == N);
        for (int i = 0; i != N; ++i)
            p[i] = v[i];
    }
    vec(std::initializer_list<double> const &il) {
        assert(il.size() == N);
        size_t i = 0;
        for (double d : il)
            p[i++] = d;
    }
    size_t size() const { return N; }

    double& operator[](size_t i) { return p[i]; }
    const double& operator[](size_t i) const { return p[i]; }

    using iterator = double*;
    using const_iterator = const double*;
    iterator begin() { return p; }
    const_iterator begin() const { return p; }
    iterator end() { return p+N; }
    const_iterator end() const { return p+N; }
};

using vec3d = vec<3>;
using vec2d = vec<2>;

template<typename T>
T vec_normalize(const T& p, const T& minref, const T& maxref)
{
    const size_t ndims = p.size();
    T rv(ndims, 0.0);
    for (size_t i = 0; i != ndims; ++i)
        rv[i] = (p[i] - minref[i])/(maxref[i] - minref[i]);
    return rv;
}

template<typename T>
T vec_min(T const& p,
          T const& q)
{
    T rv(p.size());
    for (size_t i = 0; i != rv.size(); ++i)
        rv[i] = std::min(p[i], q[i]);
    return rv;
}

template<typename T>
vec2d vec_project(T const& v3, int d1, int d2)
{
    vec2d v2;
    v2[0] = v3[d1];
    v2[1] = v3[d2];
    return v2;
}

template<typename T>
vec2d vec_project_out(T const& v3, int d1)
{
    vec2d v2;
    int j = 0;
    for (size_t i = 0, j = 0; i != v3.size(); ++i) {
        if (i != d1)
            v2[j++] = v3[i];
    }
    return v2;
}

double vec2d_area(vec2d v1, vec2d v2)
{
    return fabs(v1[0] - v2[0])*fabs(v1[1] - v2[1]);
}

template<int N>
vec<N> operator-(vec<N> v1, vec<N> v2)
{
    vec<N> r;
    for(int i = 0; i != N; ++i)
        r[i] = v1[i] - v2[i];
    return r;
}

template <int N>
struct std::formatter<vec<N>> {
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

    auto format(const vec<N>& v, std::format_context& ctx) const {
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


struct perfpoint {
    // with N objectives, each of these has N elements, each giving
    // the N weights and N corresponding performances
    std::vector<weight_v> vertices;
    std::vector<performance_v> performances;

    performance_v min_observed, max_observed;

    int id;
    static int next_id;

    mutable double cached_area{0.0};
    mutable bool have_cached_area{false};

    perfpoint() : id(next_id++) {}
    perfpoint(std::vector<weight_v> const& v,
              std::vector<performance_v> const& p,
              performance_v const& minobs,
              performance_v const& maxobs) :
        vertices(v), performances(p),
        min_observed(minobs), max_observed(maxobs),
        id(next_id++)
    {
    }

    enum describe_t { shush, describe };
    double area(describe_t descr = shush) const {
        if (have_cached_area)
            return cached_area;

        size_t ndims = performances[0].size();
        size_t realdims = std::ranges::count(global_config.types,
                                             prob_config::MINAREA);
        // it's OK for global_config to have more config that what
        // we're solving
        realdims = std::min(ndims, realdims);
        if (realdims != performances.size()) {
            throw std::runtime_error(
                std::format("Must have one performance per dimension in area computation: {} dims, {} performances",
                            realdims, performances.size()));
        }
#if 1
        switch(realdims) {
        case 2:
            cached_area = area2d(descr);
            break;
        case 3:
            //cached_area = area3d(descr);
            cached_area = area3d_max_project(descr);
            break;
        default:
            throw std::runtime_error(std::format("What? ndims = {}", ndims));
        }
#else
        cached_area = area2d(descr);
#endif
        //have_cached_area = true;
        return cached_area;
    }

    // performance_v normalize(const performance_v& p) const
    // {
    //     const size_t ndims = p.size();
    //     performance_v rv(ndims, 0.0);
    //     for (size_t i = 0; i != ndims; ++i)
    //         rv[i] = (p[i] - min_observed[i])/(max_observed[i] - min_observed[i]);
    //     return rv;
    // }

    performance_v normalize(const performance_v& p) const
    {
        return vec_normalize(p, min_observed, max_observed);
    }

    double area2d(describe_t descr) const
    {
        const size_t ndims = performances[0].size();
        double rv{1.0};
        if (descr == describe) {
            std::vector<performance_v> norm;
            for(auto& perf : performances)
                norm.push_back(normalize(perf));
            std::println("2d Computing hv of {}", performances);
            std::println("Normalized {}", norm);
        }

        // just a triangle
        for (size_t i = 0; i != ndims; ++i) {
            if (global_config.types[i] != prob_config::MINAREA)
                continue;
            double minp = performances[0][i],
                maxp = performances[0][i];
            for (size_t j = 1; j != performances.size(); ++j) {
                minp = std::min(minp, performances[j][i]);
                maxp = std::max(maxp, performances[j][i]);
            }
            double maxnorm = (maxp - min_observed[i])/(max_observed[i] - min_observed[i]);
            double minnorm = (minp - min_observed[i])/(max_observed[i] - min_observed[i]);
            if (descr) {
                std::println("Dim #{} axis size {}", i, maxnorm - minnorm);
            }
            rv *= maxnorm - minnorm;
        }
        // no reason to divide by 2
        return rv;
    }

    mutable std::vector<double> area3d_cache;

    double area3d(describe_t descr) const
    {
        const int ndims = 3;
        int realdims = std::ranges::count(global_config.types,
                                          prob_config::MINAREA);

        if (descr == describe) {
            std::vector<performance_v> norm;
            for(auto& perf : performances)
                norm.push_back(normalize(perf));
            std::println("Computing hv of {}", performances);
            std::println("Normalized {}", norm);
        }

        auto& data = area3d_cache;
        data.clear();
        double minpoint[ndims];
        for (size_t i = 0; i != performances.size(); ++i) {
            performance_v p = normalize(performances[i]);
            for (int d = 0; d != ndims; ++d) {
                if (global_config.types[d] == prob_config::FIXED)
                    continue;
                data.push_back(p[d]);
                if (i == 0)
                    minpoint[d] = p[d];
                else
                    minpoint[d] = std::min(minpoint[d], p[d]);
            }
        }
        for (int d = 0; d != ndims; ++d) {
            if (global_config.types[d] == prob_config::FIXED)
                continue;
            data.push_back(minpoint[d]);
        }
        if (descr == describe)
            std::println("realdims = {}, data = {}", realdims, data);
        // double data[] = {p0[0], p0[1], p0[2],
        //                  p1[0], p1[1], p1[2],
        //                  p2[0], p2[1], p2[2],
        //                  std::min({p0[0], p1[0], p2[0]}),
        //                  std::min({p0[1], p1[1], p2[1]}),
        //                  std::min({p0[2], p1[2], p2[2]})};

        qh::Qhull q("", realdims, performances.size()+1, data.data(), "FA Pp");

        if (descr == describe)
            std::println("Computed volume {}", q.volume());

        return q.volume();
    }

    double area3d_hv(describe_t descr) const
    {
        if (descr == describe) {
            std::vector<performance_v> norm;
            for(auto& perf : performances)
                norm.push_back(normalize(perf));
            std::println("Computing hv of {}", performances);
            std::println("Normalized {}", norm);
        }

        vec3d v0{performances[0]};
        vec3d v1{performances[0]};
        vec3d v2{performances[0]};
        vec3d minr{min_observed}, maxr{max_observed};
        auto p0 = vec_normalize(v0, minr, maxr);
        auto p1 = vec_normalize(v1, minr, maxr);
        auto p2 = vec_normalize(v2, minr, maxr);

        double ref[] = {2,2,2};
        double data_cur[] = { p0[0], p0[1], p0[2],
                              p1[0], p1[1], p1[2],
                              p2[0], p2[1], p2[2]};

        auto p01 = vec_min(p0, p1);
        auto p02 = vec_min(p0, p2);
        auto p12 = vec_min(p1, p2);

        // potential improvement
        double data_pot[] = { p0[0], p0[1], p0[2],
                              p1[0], p1[1], p1[2],
                              p2[0], p2[1], p2[2],
                              p01[0], p01[1], p01[2],
                              p02[0], p02[1], p02[2],
                              p12[0], p12[1], p12[2]
        };

        if (descr == describe) {
            std::println("mins: p01 = {}, p02 = {}, p12 = {}", p01, p02, p12);
        }

        double curarea = fpli_hv(data_cur, 3, 3, ref);
        double potarea = fpli_hv(data_pot, 3, 6, ref);

        return potarea - curarea;
    }

    auto area3d_projections(describe_t descr) const ->
        std::tuple<double, double, double>
    {
        if (descr == describe) {
            std::vector<performance_v> norm;
            for(auto& perf : performances)
                norm.push_back(normalize(perf));
            std::println("Computing hv of {}", performances);
            std::println("Normalized {}", norm);
        }

        vec3d v0{performances[0]};
        vec3d v1{performances[1]};
        vec3d v2{performances[2]};
        vec3d minr{min_observed}, maxr{max_observed};
        auto p0 = vec_normalize(v0, minr, maxr);
        auto p1 = vec_normalize(v1, minr, maxr);
        auto p2 = vec_normalize(v2, minr, maxr);

        auto p0_01 = vec_project(p0, 0, 1);
        auto p0_02 = vec_project(p0, 0, 2);
        auto p0_12 = vec_project(p0, 1, 2);

        auto p1_01 = vec_project(p1, 0, 1);
        auto p1_02 = vec_project(p1, 0, 2);
        auto p1_12 = vec_project(p1, 1, 2);

        auto p2_01 = vec_project(p2, 0, 1);
        auto p2_02 = vec_project(p2, 0, 2);
        auto p2_12 = vec_project(p2, 1, 2);

        double area01 =
            std::max({vec2d_area(p0_01, p1_01),
                    vec2d_area(p0_02, p1_02),
                    vec2d_area(p0_12, p1_12)});
        double area02 =
            std::max({vec2d_area(p0_01, p2_01),
                    vec2d_area(p0_02, p2_02),
                    vec2d_area(p0_12, p2_12)});
        double area12 =
            std::max({vec2d_area(p1_01, p2_01),
                    vec2d_area(p1_02, p2_02),
                    vec2d_area(p1_12, p2_12)});

        if (descr == describe) {
            std::println("area 01: {} 02: {} 12: {}",
                         area01, area02, area12);
        }

        // double dir01 =
        //     vec2d_area(p0_01, p1_01) +
        //     vec2d_area(p0_02, p1_02) +
        //     vec2d_area(p0_12, p1_12);
        // double dir02 =
        //     vec2d_area(p0_01, p2_01) +
        //     vec2d_area(p0_02, p2_02) +
        //     vec2d_area(p0_12, p2_12);
        // double dir12 =
        //     vec2d_area(p1_01, p2_01) +
        //     vec2d_area(p1_02, p2_02) +
        //     vec2d_area(p1_12, p2_12);

        return std::tuple{area01,area02,area12};
    }

    double area3d_max_project(describe_t descr) const
    {
        auto [a01, a02, a12] = area3d_projections(descr);
        return a01+a02+a12;
    }

    std::pair<int, int> direction_project()
    {
        auto [a01, a02, a12] = area3d_projections(shush);
        if (a01 > a02 && a01 > a12) {
            return std::pair{0,1};
        } else if (a02 > a12) {
            return std::pair{0,2};
        } else
            return std::pair{1,2};
    }

    void describe_area() const
    {
        area(describe);
    }
};

int perfpoint::next_id{0};

struct perfpoint_comp {
    bool operator()(perfpoint const& a, perfpoint const& b) const
    {
        return std::make_tuple(a.area(), -a.id) < std::make_tuple(b.area(), -b.id);
    }
};

bool intersects_bounds(perfpoint& p, prob_config& cfg)
{
    const size_t ndims = p.min_observed.size();
    for (size_t i = 0; i != ndims; ++i) {
        if (!cfg.hasbounds[i])
            continue;
        double minperf = p.max_observed[i], maxperf = p.min_observed[i];
        for (size_t j = 0; j != p.performances.size(); ++j) {
            minperf = std::min(minperf, p.performances[j][i]);
            maxperf = std::max(maxperf, p.performances[j][i]);
        }

        if (minperf > cfg.ubound[i] || maxperf < cfg.lbound[i])
            return false;
    }
    return true;
}

// solve a problem which is expected to be near the average of probs
std::pair<
    std::unique_ptr<lpprob>,
    weight_v>
solve_expected_average(cplexenv& cpx,
                       std::vector<std::unique_ptr<lpprob>> const& probs,
                       perfpoint const& p)
{
    bool solved{false};

    const int ndims = probs.size();
    std::unique_ptr<lpprob> scaled;
    weight_v newv(probs.size(), 0.0);

    bool first{true};
    do {
        // get a random vertex near the average of all the weights to
        // make the new scaled problem. We try the exact average
        // first, then if that fails we get a random point near it in
        // order to avoid cplex's temper tantrums
        std::normal_distribution d{1.0/p.vertices.size(), 0.1};
        std::fill(begin(newv), end(newv), 0.0);

        if (first) {
            for (int i = 0; i != ndims; ++i) {
                for (auto& v : p.vertices)
                    newv[i] += v[i];
                newv[i] /= p.vertices.size();
            }
            first = false;
        } else {
            newv = p.vertices[0];
            for (size_t i = 1; i != p.vertices.size(); ++i) {
                auto a = std::clamp(d(cpx.rng), 0.0, 1.0);
                std::println("Vertex {}, a = {}", i, a);
                for (int j = 0; j != ndims; ++j) {
                    newv[j] += (p.vertices[i][j] - p.vertices[0][j]) * a;
                }
            }
        }
        std::println("Trying new vertex {}", newv);
        auto scaled_tmp = make_scaled(cpx, probs, newv);
        try {
            scaled_tmp->solve();
            solved = true;
            scaled = std::move(scaled_tmp);
        } catch(std::exception& e) {
            std::println("Abort attempt, runtime error {}", e.what());
        }
    } while(!solved);
    return std::pair{std::move(scaled), newv};
}

void explore_pareto_minarea(cplexenv& cpx,
                            std::vector<std::unique_ptr<lpprob>> const& probs,
                            FILE *objectives,
                            FILE *solutions)
{
    auto [all_sobj,
          min_observed, max_observed] = get_extremes(cpx, probs, objectives,
                                                     solutions);

    const int ndims = probs.size();
    // int realdims = std::ranges::count(global_config.types,
    //                                   prob_config::MINAREA);

    std::vector<weight_v> init_vertices;
    for (int i = 0; i != ndims; ++i) {
        if (global_config.types[i] == prob_config::FIXED)
            continue;
        weight_v wi(ndims, 0);
        wi[i] = 1.0;
        for(int j = 0; j != ndims; ++j) {
            if (global_config.types[j] == prob_config::FIXED)
                wi[j] = global_config.coeffs[j];
        }
        init_vertices.emplace_back(std::move(wi));
    }

    std::vector<performance_v> init_perfs;
    for (size_t i = 0; i != all_sobj.size(); ++i)
        if (global_config.types[i] != prob_config::FIXED)
            init_perfs.push_back(all_sobj[i]);

    std::priority_queue<perfpoint, std::vector<perfpoint>,
                        perfpoint_comp> Q;
    perfpoint initppoint{init_vertices, init_perfs, min_observed, max_observed};
    Q.push(initppoint);
    std::println("Initial performances {}", init_perfs);
    std::println("Initial point area {}", initppoint.area());
    initppoint.describe_area();

    int points{0};
    while(!Q.empty() && points < nsamples) {
        ++points;

        auto p = Q.top();
        Q.pop();

        std::println("--------------------------------------------------");
        std::println("Perfpoint with area {}", p.area());
        std::println("Combining vertices {}", p.vertices);

        // vec3d origin{1,0,0};
        // vec3d direction{0,1.0/ndims, 1.0/ndims};
        // if (ndims == 3 && project_move) {
        //     origin = {0,0,0};
        //     direction = {0,0,0};
        //     auto dir = p.direction_project();
        //     origin[dir.first] = 1.0;
        //     direction[dir.second] = 0.5;
        // }

        auto [scaled, newv] = solve_expected_average(cpx, probs, p);

        auto psol = scaled->solution();
        std::println("Scaled optimum {}", scaled->solution_cost());
        auto pfpoint = probs
            | std::ranges::views::transform([&](auto&& q){
                return q->evaluate(psol); })
            | std::ranges::to<std::vector>();
        std::println("Point #{}: {}", points, pfpoint);
        std::println(objectives, "{:c}", pfpoint);
        std::println(solutions, "{:c}", psol);
        fflush(objectives);
        fflush(solutions);

        // copy the vertices of the perfpoint to generate all the new
        // combinations of vertices and performances
        std::println("{} vertices, should create {} new perfpoints",
                     p.vertices.size(), p.vertices.size());
        auto vs = p.vertices;
        auto perfs = p.performances;
        int newpoints{0};
        for (size_t i = 0; i != vs.size(); ++i) {
            vs[i] = newv;
            perfs[i] = pfpoint;
            perfpoint newp{vs, perfs, min_observed, max_observed};
            Q.push(newp);
            std::println("New perfpoint: vertices {}\n"
                         "\tperformances {}\n"
                         "\tarea {}",
                         vs, perfs,
                         newp.area());
            newp.describe_area();
            if (p.area() < newp.area())
                std::println("************* WARNING: new point has larger area"
                             "*************\n"
                             "\t{} > {}", newp.area(), p.area());
            vs[i] = p.vertices[i];
            perfs[i] = p.performances[i];
            ++newpoints;
        }
        std::println("Created {} new points", newpoints);
    }
}

// generate vertices using the minarea algorithm, but instead of
// generating (x,y), generate (x,y,0) and (x,y,1), with the extra
// dimension corresponding to nextprob, if it is non-null
std::vector<perfpoint>
explore_pareto_seeded(cplexenv& cpx,
                      std::vector<std::unique_ptr<lpprob>> const& probs,
                      std::vector<perfpoint> const& seeds,
                      performance_v const& min_observed,
                      performance_v const& max_observed,
                      int* pointcounter,
                      FILE *objectives,
                      FILE *solutions)
{
    int points{0};
    std::priority_queue<perfpoint, std::vector<perfpoint>,
                        perfpoint_comp> Q;

    for (auto p : seeds)
        Q.push(p);
    while(!Q.empty() && points < nsamples) {
        ++points;

        auto p = Q.top();
        Q.pop();

        std::println("--------------------------------------------------");
        std::println("Perfpoint with area {}", p.area());
        std::println("Combining vertices {}", p.vertices);

        auto [scaled, newv] = solve_expected_average(cpx, probs, p);

        auto psol = scaled->solution();
        std::println("Scaled optimum {}", scaled->solution_cost());
        auto pfpoint = probs
            | std::ranges::views::transform([&](auto&& q){
                return q->evaluate(psol); })
            | std::ranges::to<std::vector>();
        std::println("Point #{} (#{}): {}", points, *pointcounter, pfpoint);
        if (objectives) {
            std::println(objectives, "{:c}", pfpoint);
            std::println(solutions, "{:c}", psol);
            fflush(objectives);
            fflush(solutions);
            ++(*pointcounter);
        }

        // copy the vertices of the perfpoint to generate all the new
        // combinations of vertices and performances
        std::println("{} vertices, should create {} new perfpoints",
                     p.vertices.size(), p.vertices.size());
        auto vs = p.vertices;
        auto perfs = p.performances;
        int newpoints{0};
        for (size_t i = 0; i != vs.size(); ++i) {
            vs[i] = newv;
            perfs[i] = pfpoint;
            perfpoint newp{vs, perfs, min_observed, max_observed};
            if (!intersects_bounds(newp, global_config)) {
                std::println("Rejecting perfpoint: vertices {}\n"
                             "\tperformances {}\n",
                             vs, perfs);
                --points;
            } else {
                Q.push(newp);
                std::println("New perfpoint: vertices {}\n"
                             "\tperformances {}\n"
                             "\tarea {}",
                             vs, perfs,
                             newp.area());
                newp.describe_area();
                if (p.area() < newp.area())
                    std::println("************* WARNING: new point has larger area"
                                 "*************\n"
                                 "\t{} > {}", newp.area(), p.area());
                ++newpoints;
            }
            vs[i] = p.vertices[i];
            perfs[i] = p.performances[i];
        }
        std::println("Created {} new points", newpoints);
    }

    std::vector<perfpoint> rv;
    while (!Q.empty()) {
        rv.push_back(Q.top());
        Q.pop();
    }
    return rv;
}

void explore_pareto_staged(cplexenv& cpx,
                           std::vector<std::unique_ptr<lpprob>>& probs,
                           FILE *objectives,
                           FILE *solutions)
{
    auto [all_sobj,
          min_observed, max_observed] = get_extremes(cpx, probs, objectives,
                                                     solutions);

    // start with 2 dimensions
    std::vector<std::unique_ptr<lpprob>> remaining_probs;
    while(probs.size() > 2) {
        remaining_probs.push_back(std::move(probs.back()));
        probs.pop_back();
    }

    std::vector<weight_v> init_vertices;
    init_vertices.push_back({1.0, 0.0});
    init_vertices.push_back({0.0, 1.0});

    // shrink it all down to 2 dimensions
    std::vector<performance_v> init_perfs(all_sobj);
    init_perfs.resize(2);
    for(auto& p : init_perfs)
        p.resize(2);

    std::vector<perfpoint> seeds;
    perfpoint initppoint{init_vertices, init_perfs, min_observed, max_observed};
    seeds.push_back(initppoint);
    std::println("Initial performances {}", init_perfs);
    std::println("Initial point area {}", initppoint.area());
    initppoint.describe_area();

    int iteration{0};
    int pointcounter{0};

    while(true) {
        global_config = config_stages[iteration];
        ++iteration;

        std::println("//////////////////////////////////////////////////");
        std::println("--------------------------------------------------");
        std::println("//////////////////////////////////////////////////");
        std::println("Solving with {} dimensions", 1+iteration);

        if (remaining_probs.empty()) {
            for(auto point : seeds) {
                std::vector<perfpoint> singleseed({point});
                explore_pareto_seeded(cpx, probs, singleseed,
                                      min_observed, max_observed,
                                      &pointcounter,
                                      objectives, solutions);
            }
            break;
        }
        std::vector<perfpoint> finalpoints;
        for(auto point : seeds) {
            std::vector<perfpoint> singleseed({point});
            auto ns = explore_pareto_seeded(cpx, probs, singleseed,
                                            min_observed, max_observed,
                                            &pointcounter,
                                            nullptr, nullptr);
            finalpoints.insert(finalpoints.end(), ns.begin(), ns.end());
        }
        seeds.clear();

        probs.push_back(std::move(remaining_probs.back()));
        remaining_probs.pop_back();

        // which problem remains MINAREA in the next iteration
        auto& nextconf = config_stages[iteration];
        auto nactive = std::count(nextconf.types.begin(), nextconf.types.begin()+iteration+1,
                                 prob_config::MINAREA);
        if ( nactive != 1) {
            throw std::runtime_error("Exactly one MINAREA problem must be kept");
        }
        auto active_idx = std::find(nextconf.types.begin(), nextconf.types.begin()+iteration+1,
                                    prob_config::MINAREA) - nextconf.types.begin();

        size_t counter{0};
        for (perfpoint& point : finalpoints) {
            ++counter;
            std::println("//////////////////////////////////////////////////");
            std::println("Extending perfpoint {}/{} @vertices {} to next dimension",
                         counter, finalpoints.size(),
                         point.vertices);
            weight_v vx0 = point.vertices[0];
            weight_v vx1 = point.vertices[1];

            vec2d operf0{point.performances[0]};
            vec2d operf1{point.performances[1]};

            if (operf0[active_idx] > operf1[active_idx])
                swap(vx0, vx1);

            // recompute (x,y) to get (x,y,0). We have to temporarily
            // remove the new problem from 'probs'
            auto tmpptr = std::move(probs.back());
            probs.pop_back();
            auto scaled_0 = make_scaled(cpx, probs, vx0);
            // back in
            probs.push_back(std::move(tmpptr));
            vx0.push_back(0);
            scaled_0->solve(); // this will succeed because it has
                             // succeeded before
            auto psol0 = scaled_0->solution();
            auto perf0 = probs
                | std::ranges::views::transform([&](auto&& q){
                    return q->evaluate(psol0); })
                | std::ranges::to<std::vector>();

            std::println("Got vertex {}, performance {}", vx0, perf0);
            if (remaining_probs.empty()) {
                std::println(objectives, "{:c}", perf0);
                std::println(solutions, "{:c}", psol0);
                fflush(objectives);
                fflush(solutions);
                ++pointcounter;
            }

            // now do (x,y,1)
            vx1.push_back(config_stages[iteration].coeffs[iteration+1]);
            auto scaled_1 = make_scaled(cpx, probs, vx1);
            scaled_1->solve();
            auto psol1 = scaled_1->solution();
            auto perf1 = probs
                | std::ranges::views::transform([&](auto&& q){
                    return q->evaluate(psol1); })
                | std::ranges::to<std::vector>();
            std::println("Got vertex {}, performance {}", vx1, perf1);
            perfpoint pfp{{vx0, vx1}, {perf0, perf1}, min_observed, max_observed};
            seeds.push_back(pfp);
            if (remaining_probs.empty()) {
                std::println(objectives, "{:c}", perf1);
                std::println(solutions, "{:c}", psol1);
                fflush(objectives);
                fflush(solutions);
                ++pointcounter;
            }
        }
    }
}

void explore_pareto(cplexenv& cpx,
                    std::vector<std::unique_ptr<lpprob>>& probs,
                    FILE *objectives,
                    FILE *solutions)
{
    //explore_pareto_weighted_uniform(cpx, probs, objectives, solutions);
    //explore_pareto_weighted_adaptive(cpx, probs, objectives, solutions);
    if (probs.size() == 2)
        explore_pareto_minarea(cpx, probs, objectives, solutions);
    else
        explore_pareto_staged(cpx, probs, objectives, solutions);
}

int main(int argc, char **argv)
{
    cplexenv env;

    std::list<std::string> args(argv+1, argv+argc);
    std::vector<std::unique_ptr<lpprob>> probs;
    for(auto&& s: args) {
        std::println("Reading \"{}\"", s);
        probs.push_back(std::make_unique<lpprob>(env));
        auto& lp = *probs.back();
        lp.read(s);
    }

    FILE *objectives = std::fopen("objectives.csv", "w");
    FILE *solutions = std::fopen("solutions.csv", "w");
    auto names = probs
        | std::ranges::views::transform([&](auto&& p) { return p->readfrom; })
        | std::ranges::to<std::vector>();
    std::println(objectives, "{:c}", names);
    explore_pareto(env, probs, objectives, solutions);
    std::fclose(objectives);
    std::fclose(solutions);
}
