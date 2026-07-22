#pragma once

// Deterministic data parallelism for per-item independent work.
//
// Every helper here distributes DISJOINT index ranges over threads; each
// item's computation is untouched (same operations, same order within the
// item), so results are bit-identical to the serial loop.  Never use these
// for reductions whose floating-point result depends on summation order.

#include <algorithm>
#include <future>
#include <thread>
#include <vector>

namespace hydramarker {

// Run fn(begin, end) over [0, n) split into chunks across threads.
// fn must only write to per-index locations (disjoint between chunks).
// Serial fallback for small n where thread dispatch would dominate.
template <typename Fn>
void parallelChunks(int n, int min_parallel_n, Fn&& fn) {
    if (n <= 0) return;

    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    const int max_chunks = std::max(1, std::min(hw, 8));

    if (n < min_parallel_n || max_chunks <= 1) {
        fn(0, n);
        return;
    }

    const int n_chunks = std::min(max_chunks, n);
    const int chunk = (n + n_chunks - 1) / n_chunks;

    std::vector<std::future<void>> futures;
    futures.reserve(static_cast<size_t>(n_chunks) - 1);

    for (int c = 1; c < n_chunks; ++c) {
        const int begin = c * chunk;
        const int end = std::min(n, begin + chunk);
        if (begin >= end) break;
        futures.emplace_back(std::async(
            std::launch::async,
            [begin, end, &fn]() { fn(begin, end); }));
    }

    fn(0, std::min(n, chunk));

    for (auto& f : futures) {
        f.get();   // propagates exceptions from worker chunks
    }
}

}  // namespace hydramarker
