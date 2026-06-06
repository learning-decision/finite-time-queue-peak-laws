// Native fast path for the qpeak parallel_server ring experiment.
//
// This translation unit intentionally has no Python/Numpy dependency.  Python passes
// raw pointers via ctypes; the code fills small output arrays and returns.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct SplitMix64 {
    uint64_t state;

    explicit SplitMix64(uint64_t seed) : state(seed) {}

    uint64_t next_u64() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    double uniform01() {
        // 53 random bits, exactly representable in double mantissa.
        return static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }

    uint64_t bounded(uint64_t n) {
        if (n <= 1) return 0;
        // Rejection sampling, unbiased for arbitrary n.
        const uint64_t threshold = static_cast<uint64_t>(-n) % n;
        while (true) {
            const uint64_t x = next_u64();
            if (x >= threshold) return x % n;
        }
    }
};

static inline int popcount64(uint64_t x) {
#if defined(__GNUG__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    int c = 0;
    while (x) { x &= (x - 1); ++c; }
    return c;
#endif
}

static inline int ctz64(uint64_t x) {
#if defined(__GNUG__) || defined(__clang__)
    return __builtin_ctzll(x);
#else
    int c = 0;
    while ((x & 1ull) == 0ull) { x >>= 1; ++c; }
    return c;
#endif
}

static inline int select_nth_set_bit(uint64_t mask, int rank) {
    // rank is 0-based among set bits. K<=63 in the caller, so a tiny loop is fine.
    while (rank > 0) {
        mask &= (mask - 1ull);
        --rank;
    }
    return ctz64(mask);
}

struct BinomTables {
    int nmax;
    double cdf[65][66];

    void init(int nmax_, double p) {
        nmax = nmax_;
        for (int n = 0; n <= 64; ++n) {
            for (int k = 0; k <= 65; ++k) cdf[n][k] = 1.0;
        }
        for (int n = 0; n <= nmax; ++n) {
            long double pk;
            if (p == 1.0) {
                for (int k = 0; k < n; ++k) cdf[n][k] = 0.0;
                cdf[n][n] = 1.0;
                continue;
            }
            if (p == 0.0) {
                cdf[n][0] = 1.0;
                for (int k = 1; k <= n; ++k) cdf[n][k] = 1.0;
                continue;
            }
            pk = std::pow(1.0L - static_cast<long double>(p), n);
            long double acc = 0.0L;
            for (int k = 0; k <= n; ++k) {
                acc += pk;
                cdf[n][k] = static_cast<double>(acc > 1.0L ? 1.0L : acc);
                if (k < n) {
                    pk *= (static_cast<long double>(n - k) / static_cast<long double>(k + 1));
                    pk *= (static_cast<long double>(p) / (1.0L - static_cast<long double>(p)));
                }
            }
            cdf[n][n] = 1.0;
        }
    }

    int sample(int n, SplitMix64 &rng) const {
        if (n <= 0) return 0;
        const double u = rng.uniform01();
        for (int k = 0; k <= n; ++k) {
            if (u <= cdf[n][k]) return k;
        }
        return n;
    }
};

static inline uint64_t uniform_subset_mask_full(int n, int m, SplitMix64 &rng) {
    if (m <= 0) return 0ull;
    if (m >= n) return (n == 64) ? ~0ull : ((1ull << n) - 1ull);
    uint64_t selected = 0ull;
    // Floyd's algorithm: uniform m-subset of {0,...,n-1}.
    for (int j = n - m; j < n; ++j) {
        const int t = static_cast<int>(rng.bounded(static_cast<uint64_t>(j + 1)));
        const uint64_t bit_t = 1ull << t;
        if (selected & bit_t) {
            selected |= (1ull << j);
        } else {
            selected |= bit_t;
        }
    }
    return selected;
}

static inline uint64_t uniform_subset_of_mask(uint64_t source_mask, int source_count, int m, SplitMix64 &rng) {
    if (m <= 0 || source_mask == 0ull) return 0ull;
    if (m >= source_count) return source_mask;
    const uint64_t rank_mask = uniform_subset_mask_full(source_count, m, rng);
    uint64_t out = 0ull;
    uint64_t ranks = rank_mask;
    while (ranks) {
        const int rank = ctz64(ranks);
        ranks &= (ranks - 1ull);
        const int idx = select_nth_set_bit(source_mask, rank);
        out |= (1ull << idx);
    }
    return out;
}

static inline void update_q(int64_t *q, int cls, int delta, int64_t &total, long double &sumsq) {
    const int64_t oldv = q[cls];
    const int64_t newv = oldv + static_cast<int64_t>(delta);
    q[cls] = newv;
    total += static_cast<int64_t>(delta);
    sumsq += static_cast<long double>(newv) * static_cast<long double>(newv)
           - static_cast<long double>(oldv) * static_cast<long double>(oldv);
}

static inline int action_feasible(int action, int k, int L, const int64_t *q, uint64_t idle_mask) {
    if (action == 0) return 1;
    if (((idle_mask >> k) & 1ull) == 0ull) return 0;
    if (action == 1) {
        return q[k] > 0;
    }
    // action == 2: server k serves class k-1 (cyclic).
    const int cls = (k == 0) ? (L - 1) : (k - 1);
    return q[cls] > 0;
}

static inline double action_weight(int action, int k, int L, const int64_t *q) {
    if (action == 0) return 0.0;
    if (action == 1) return static_cast<double>(q[k]);
    const int cls = (k == 0) ? (L - 1) : (k - 1);
    return static_cast<double>(q[cls]);
}

static inline int class_count_ok(int cls, int L, const int64_t *q, int action_left_server, int action_right_server) {
    // Class cls can be served by server cls using action 1 and server cls+1 using action 2.
    int count = 0;
    if (action_left_server == 1) ++count;
    if (action_right_server == 2) ++count;
    return count <= q[cls];
}

void assign_ring_dp_exact(
    int L,
    const int64_t *q,
    uint64_t idle_mask,
    int *actions
) {
    // Exact dynamic program for the cycle with server actions:
    //   0 = do not assign, 1 = serve class k, 2 = serve class k-1.
    // Constraint for each class l:
    //   1{a_l=1} + 1{a_{l+1}=2} <= q_l.
    if (L <= 0) return;
    if (L == 1) {
        actions[0] = (idle_mask & 1ull) && q[0] > 0 ? 1 : 0;
        return;
    }

    const double NEG = -1.0e100;
    double best_value = NEG;
    int best_last = 0;
    int best_parent[64][3];
    int best_a0 = 0;

    for (int a0 = 0; a0 < 3; ++a0) {
        if (!action_feasible(a0, 0, L, q, idle_mask)) continue;

        double prev[3] = {NEG, NEG, NEG};
        double cur[3] = {NEG, NEG, NEG};
        int parent[64][3];
        for (int i = 0; i < 64; ++i) for (int j = 0; j < 3; ++j) parent[i][j] = 0;
        prev[a0] = action_weight(a0, 0, L, q);

        for (int k = 1; k < L; ++k) {
            cur[0] = cur[1] = cur[2] = NEG;
            for (int a = 0; a < 3; ++a) {
                if (!action_feasible(a, k, L, q, idle_mask)) continue;
                for (int p = 0; p < 3; ++p) {
                    if (prev[p] <= NEG / 2) continue;
                    const int cls = k - 1;
                    if (!class_count_ok(cls, L, q, p, a)) continue;
                    const double val = prev[p] + action_weight(a, k, L, q);
                    if (val > cur[a]) {
                        cur[a] = val;
                        parent[k][a] = p;
                    }
                }
            }
            prev[0] = cur[0]; prev[1] = cur[1]; prev[2] = cur[2];
        }

        for (int last = 0; last < 3; ++last) {
            if (prev[last] <= NEG / 2) continue;
            if (!class_count_ok(L - 1, L, q, last, a0)) continue;
            const double val = prev[last];
            if (val > best_value) {
                best_value = val;
                best_last = last;
                best_a0 = a0;
                for (int i = 0; i < 64; ++i) for (int j = 0; j < 3; ++j) best_parent[i][j] = parent[i][j];
            }
        }
    }

    for (int k = 0; k < L; ++k) actions[k] = 0;
    if (best_value <= NEG / 2) return;
    actions[L - 1] = best_last;
    for (int k = L - 1; k >= 1; --k) {
        actions[k - 1] = best_parent[k][actions[k]];
    }
    actions[0] = best_a0;
}

void assign_and_update_ring(
    int L,
    int64_t *q,
    uint64_t &busy_mask,
    int64_t &total,
    long double &sumsq
) {
    const uint64_t full_mask = (L == 64) ? ~0ull : ((1ull << L) - 1ull);
    const uint64_t idle_mask = (~busy_mask) & full_mask;
    if (idle_mask == 0ull) return;

    int actions[64];
    for (int i = 0; i < L; ++i) actions[i] = 0;

    // Fast path: independent greedy is globally optimal whenever it does not
    // over-use a class with q_l=1.  In heavy traffic q_l is usually >=2, so the
    // exact DP below is rarely needed after warm-up.
    bool conflict = false;
    uint64_t used_q1 = 0ull;
    uint64_t servers = idle_mask;
    while (servers) {
        const int k = ctz64(servers);
        servers &= (servers - 1ull);
        const int left_cls = (k == 0) ? (L - 1) : (k - 1);
        const int right_cls = k;
        const int64_t q_left = q[left_cls];
        const int64_t q_right = q[right_cls];
        int cls = -1;
        int action = 0;
        if (q_right <= 0 && q_left <= 0) {
            action = 0;
        } else if (q_right >= q_left) {
            // Deterministic tie break toward class k.
            cls = right_cls;
            action = 1;
        } else {
            cls = left_cls;
            action = 2;
        }
        actions[k] = action;
        if (cls >= 0 && q[cls] == 1) {
            const uint64_t bit = 1ull << cls;
            if (used_q1 & bit) {
                conflict = true;
                break;
            }
            used_q1 |= bit;
        }
    }

    if (conflict) {
        assign_ring_dp_exact(L, q, idle_mask, actions);
    }

    for (int k = 0; k < L; ++k) {
        const int action = actions[k];
        if (action == 0) continue;
        const int cls = (action == 1) ? k : ((k == 0) ? (L - 1) : (k - 1));
        // Defensive guard; exact/greedy assignment should already guarantee this.
        if (((idle_mask >> k) & 1ull) && q[cls] > 0) {
            busy_mask |= (1ull << k);
            update_q(q, cls, -1, total, sumsq);
        }
    }
}

static inline double series_value(int code, int64_t total, long double sumsq, int64_t peak_total, long double peak_sumsq) {
    switch (code) {
        case 0: // peak_l1_so_far
        case 5: // peak_totalQ_so_far
            return static_cast<double>(peak_total);
        case 1: // peak_l2_so_far
            return std::sqrt(static_cast<double>(peak_sumsq));
        case 2: // totalQ
        case 3: // q_norm_l1 for nonnegative vector queues
            return static_cast<double>(total);
        case 4: // q_norm_l2
            return std::sqrt(static_cast<double>(sumsq));
        default:
            return std::numeric_limits<double>::quiet_NaN();
    }
}

void simulate_one_rep(
    int r,
    int64_t T,
    int L,
    double lambda,
    const BinomTables &arrivals_binom,
    const BinomTables &completion_binom,
    uint64_t seed,
    const int64_t *t_indices,
    int G,
    const int *series_codes,
    int S,
    double *per_rep_out,
    int R,
    int P,
    float *q_paths,
    const int *path_series_codes,
    int PS,
    double *path_series_out,
    int progress,
    int64_t progress_every,
    std::atomic<int64_t> *done_slots,
    int64_t total_work,
    std::chrono::steady_clock::time_point start_time,
    std::atomic<int64_t> *last_progress_ms
) {
    (void)lambda; // lambda is baked into arrivals_binom; keep arg for readability.
    SplitMix64 rng(seed + 0xD1B54A32D192ED03ull * static_cast<uint64_t>(r + 1));
    int64_t q[64];
    for (int i = 0; i < 64; ++i) q[i] = 0;
    uint64_t busy_mask = 0ull;
    int64_t total = 0;
    long double sumsq = 0.0L;
    int64_t peak_total = 0;
    long double peak_sumsq = 0.0L;

    int g = 0;
    const bool save_path = (r < P && q_paths != nullptr);
    int64_t local_progress = 0;

    for (int64_t t = 0; t < T; ++t) {
        // Observe Q_t and update exact running peaks before scheduling.
        if (total > peak_total) peak_total = total;
        if (sumsq > peak_sumsq) peak_sumsq = sumsq;

        if (g < G && t == t_indices[g]) {
            for (int s = 0; s < S; ++s) {
                per_rep_out[(static_cast<int64_t>(s) * R + r) * G + g] =
                    series_value(series_codes[s], total, sumsq, peak_total, peak_sumsq);
            }
            if (save_path) {
                const int64_t base = (static_cast<int64_t>(r) * G + g) * L;
                for (int i = 0; i < L; ++i) {
                    q_paths[base + i] = static_cast<float>(q[i]);
                }
                if (path_series_out != nullptr) {
                    for (int ps = 0; ps < PS; ++ps) {
                        path_series_out[(static_cast<int64_t>(r) * PS + ps) * G + g] =
                            series_value(path_series_codes[ps], total, sumsq, peak_total, peak_sumsq);
                    }
                }
            }
            ++g;
        }

        // 1. Busy servers complete independently with probability mu.
        const int busy_count = popcount64(busy_mask);
        const int n_complete = completion_binom.sample(busy_count, rng);
        if (n_complete > 0) {
            const uint64_t complete_mask = uniform_subset_of_mask(busy_mask, busy_count, n_complete, rng);
            busy_mask &= ~complete_mask;
        }

        // 2. Assign idle servers by exact MaxWeight for the ring, updating q by -S_t.
        assign_and_update_ring(L, q, busy_mask, total, sumsq);

        // 3. Arrivals A_{t+1}: IID Bernoulli(lambda) over classes.
        const int n_arrivals = arrivals_binom.sample(L, rng);
        if (n_arrivals > 0) {
            uint64_t arr_mask = uniform_subset_mask_full(L, n_arrivals, rng);
            while (arr_mask) {
                const int cls = ctz64(arr_mask);
                arr_mask &= (arr_mask - 1ull);
                update_q(q, cls, +1, total, sumsq);
            }
        }

        if (progress && progress_every > 0) {
            ++local_progress;
            if (local_progress >= progress_every || t + 1 == T) {
                const int64_t inc = local_progress;
                local_progress = 0;
                const int64_t done = done_slots->fetch_add(inc) + inc;
                const auto now = std::chrono::steady_clock::now();
                const int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                int64_t last_ms = last_progress_ms->load();
                if (ms - last_ms >= 1000 || done >= total_work) {
                    int64_t expected = last_ms;
                    if (last_progress_ms->compare_exchange_strong(expected, ms)) {
                        const double elapsed = std::max(1e-9, static_cast<double>(ms) / 1000.0);
                        const double rate = static_cast<double>(done) / elapsed;
                        std::fprintf(stderr, "\r[fast progress] completed_slots=%lld  %.2f%%  %.3g slots/s",
                                     static_cast<long long>(done),
                                     100.0 * static_cast<double>(done) / static_cast<double>(total_work),
                                     rate);
                        std::fflush(stderr);
                    }
                }
            }
        }
    }
}

} // namespace

extern "C" int qpeak_simulate_parallel_server_ring(
    int64_t T,
    int R,
    int L,
    double lambda,
    double mu,
    uint64_t seed,
    const int64_t *t_indices,
    int G,
    const int *series_codes,
    int S,
    double *per_rep_out,
    int P,
    float *q_paths,
    const int *path_series_codes,
    int PS,
    double *path_series_out,
    int progress,
    int64_t progress_every,
    int num_threads
) {
    if (T <= 0 || R <= 0 || L <= 0 || L > 63 || G <= 0 || S <= 0) return 1;
    if (!(lambda > 0.0 && lambda < 1.0)) return 2;
    if (!(mu > 0.0 && mu <= 1.0)) return 3;
    if (t_indices == nullptr || series_codes == nullptr || per_rep_out == nullptr) return 4;

#ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#else
    (void)num_threads;
#endif

    BinomTables arrivals_binom;
    BinomTables completion_binom;
    arrivals_binom.init(L, lambda);
    completion_binom.init(L, mu);

    std::atomic<int64_t> done_slots(0);
    std::atomic<int64_t> last_progress_ms(0);
    const auto start_time = std::chrono::steady_clock::now();

#pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < R; ++r) {
        simulate_one_rep(
            r, T, L, lambda, arrivals_binom, completion_binom, seed,
            t_indices, G, series_codes, S, per_rep_out, R,
            P, q_paths, path_series_codes, PS, path_series_out,
            progress, progress_every, &done_slots, static_cast<int64_t>(T) * static_cast<int64_t>(R), start_time, &last_progress_ms
        );
    }

    if (progress) {
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
    }
    return 0;
}
