// Caching cv::MatAllocator, see mat_allocator.hpp.
#include "mat_allocator.hpp"

#include <cstddef>
#include <iterator>
#include <list>
#include <mutex>
#include <new>

#include <opencv2/core.hpp>

namespace hydramarker {

namespace {

// Smaller blocks are served cheaply by the heap's own free lists.
constexpr size_t kMinCachedBytes = 256 * 1024;

// At most this much released memory is kept; beyond it the least recently
// released blocks go back to the OS (give). One search frame alone cycles
// through about 22 large blocks, 160-190 MB (1080p float images and
// gradients, recovery variants); as the block sizes vary between frames,
// the cache fills up to this bound in practice: about +256 MB of private
// memory in steady state.
constexpr size_t kMaxCachedBytes = 256 * 1024 * 1024;

// CV_AUTOSTEP of OpenCV's C API (opencv2/core/core_c.h), the value
// StdMatAllocator::allocate compares a caller's step against.
constexpr size_t kAutoStep = 0x7fffffff;

// Rounds up to one of 8 classes per power of two (at most 12.5 % slack), so
// ROI buffers whose size drifts by a few pixels between frames share blocks.
// UMatData::size keeps the requested size; the class is recomputed from it
// on release.
size_t sizeClass(size_t bytes) {
    size_t octave = kMinCachedBytes;
    while (octave <= bytes / 2) {
        octave *= 2;
    }
    const size_t step = octave / 8;
    return (bytes + step - 1) / step * step;
}

// Blocks that are kept on release; all others are plain fastMalloc /
// fastFree blocks of the exact size.
bool isCached(size_t bytes) {
    return bytes >= kMinCachedBytes && sizeClass(bytes) <= kMaxCachedBytes;
}

// Adapted from OpenCV 4.12 (Apache-2.0), modules/core/src/matrix.cpp:
// StdMatAllocator plus a free list.
class CachingMatAllocator final : public cv::MatAllocator {
public:
    cv::UMatData* allocate(
        int dims,
        const int* sizes,
        int type,
        void* data0,
        size_t* step,
        cv::AccessFlag /*flags*/,
        cv::UMatUsageFlags /*usageFlags*/
    ) const override {
        size_t total = CV_ELEM_SIZE(type);
        for (int i = dims - 1; i >= 0; i--) {
            if (step) {
                if (data0 && step[i] != kAutoStep) {
                    CV_Assert(total <= step[i]);
                    total = step[i];
                } else {
                    step[i] = total;
                }
            }
            total *= sizes[i];
        }

        uchar* data = static_cast<uchar*>(data0);
        if (!data) {
            data = isCached(total)
                ? take(sizeClass(total))
                : static_cast<uchar*>(cv::fastMalloc(total));
        }

        cv::UMatData* u = new cv::UMatData(this);
        u->data = u->origdata = data;
        u->size = total;
        if (data0) {
            u->flags |= cv::UMatData::USER_ALLOCATED;
        }
        return u;
    }

    bool allocate(
        cv::UMatData* u,
        cv::AccessFlag /*accessFlags*/,
        cv::UMatUsageFlags /*usageFlags*/
    ) const override {
        return u != nullptr;
    }

    void deallocate(cv::UMatData* u) const override {
        if (!u) {
            return;
        }

        CV_Assert(u->urefcount == 0);
        CV_Assert(u->refcount == 0);
        if (!(u->flags & cv::UMatData::USER_ALLOCATED)) {
            if (isCached(u->size)) {
                give(u->origdata, sizeClass(u->size));
            } else {
                cv::fastFree(u->origdata);
            }
            u->origdata = nullptr;
        }
        delete u;
    }

private:
    struct Block {
        uchar* data;
        size_t bytes;
    };

    // The most recently released block of exactly this class, else a new one.
    uchar* take(size_t bytes) const {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto it = free_.begin(); it != free_.end(); ++it) {
                if (it->bytes == bytes) {
                    uchar* data = it->data;
                    held_bytes_ -= bytes;
                    free_.erase(it);
                    return data;
                }
            }
        }
        return static_cast<uchar*>(cv::fastMalloc(bytes));
    }

    // Keeps the block; beyond the bound the least recently released blocks
    // are evicted. Called from deallocate (i.e. from ~Mat), so it must not
    // throw. Everything that allocates comes first: MSVC's std::list allocates
    // its sentinel node already in the default constructor, so both lists are
    // made before the node is pushed. If any of that fails, the block is
    // simply freed. From the splice on, nodes are only moved; the frees run
    // outside the lock.
    void give(uchar* data, size_t bytes) const noexcept {
        try {
            std::list<Block> node;
            std::list<Block> evicted;
            node.push_back({data, bytes});

            {
                std::lock_guard<std::mutex> lock(mutex_);
                free_.splice(free_.begin(), node);
                held_bytes_ += bytes;
                while (held_bytes_ > kMaxCachedBytes) {
                    held_bytes_ -= free_.back().bytes;
                    evicted.splice(evicted.end(), free_, std::prev(free_.end()));
                }
            }
            for (const Block& block : evicted) {
                cv::fastFree(block.data);
            }
        } catch (const std::bad_alloc&) {
            cv::fastFree(data);  // only reachable before the splice
        }
    }

    mutable std::mutex mutex_;
    mutable std::list<Block> free_;  // front = most recently released
    mutable size_t held_bytes_ = 0;
};

}  // namespace

void installCachingMatAllocator() {
    // Intentionally leaked (see the header); the static makes repeated calls
    // a no-op.
    static CachingMatAllocator* const allocator = [] {
        auto* a = new CachingMatAllocator();
        cv::Mat::setDefaultAllocator(a);
        return a;
    }();
    (void)allocator;
}

}  // namespace hydramarker
